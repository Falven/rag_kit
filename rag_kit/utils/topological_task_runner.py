"""
Module for executing asynchronous and synchronous tasks in a topological order
based on their dependencies. It uses Python's asyncio for asynchronous tasks and
a custom thread pool executor for synchronous tasks. The results of the tasks
are stored in a nested dictionary structure based on keys specified via
decorators.
Fran Aguilera
06/07/24
"""

import asyncio
import logging
from concurrent.futures import Executor, ThreadPoolExecutor
from graphlib import TopologicalSorter
from time import sleep, time
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TPTask(BaseModel):
    name: Optional[str] = Field(default=None)
    func: Callable[..., Any]
    deps: List[str] = Field(default_factory=list)
    args: List[Any] = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    res_key: Optional[str] = Field(default=None)

    @field_validator("func")
    def validate_func(cls, v):
        if not callable(v):
            raise TypeError("func must be a callable")
        return v

    @field_validator("deps", mode="before")
    def validate_deps(cls, v):
        if not isinstance(v, list):
            raise TypeError("deps must be a list")
        if not all(isinstance(dep, str) for dep in v):
            raise TypeError("Each dependency must be a string")
        return v

    @field_validator("args", mode="before")
    def validate_args(cls, v):
        if not isinstance(v, list):
            raise TypeError("args must be a list")
        return v

    @field_validator("kwargs", mode="before")
    def validate_kwargs(cls, v: Any):
        if not isinstance(v, dict):
            raise TypeError("kwargs must be a dict")
        return v

    @field_validator("res_key")
    def validate_res_key(cls, v):
        if v is not None and not isinstance(v, str):
            raise TypeError("res_key must be a string or None")
        return v


class TopologicalTaskRunner:
    def __init__(
        self,
        executor: Optional[Executor] = None,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self._failed_tasks: Dict[str, TPTask] = {}
        self.task_results: Dict[str, Any] = {}
        self.task_errors: Dict[str, BaseException] = {}
        self.logger = logger or logging.getLogger(__name__)
        self.executor = executor if executor else ThreadPoolExecutor()
        try:
            self.loop = event_loop if event_loop else asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        self.loop.set_default_executor(self.executor)

    @classmethod
    def from_default(cls, **kwargs) -> "TopologicalTaskRunner":
        """
        Create a TopologicalTaskRunner using the default executor and event
        loop.
        """
        return cls(**kwargs)

    @classmethod
    def from_executor(cls, executor: Executor, **kwargs) -> "TopologicalTaskRunner":
        """Create a TopologicalTaskRunner with a custom executor.
        Args:
            executor (Executor): The custom executor to use.
        """
        return cls(executor=executor, **kwargs)

    @classmethod
    def from_event_loop(
        cls, loop: asyncio.AbstractEventLoop, **kwargs
    ) -> "TopologicalTaskRunner":
        """Create a TopologicalTaskRunner with a custom event loop.
        Args:
            loop (asyncio.AbstractEventLoop): The custom event loop to use.
        """
        return cls(event_loop=loop, **kwargs)

    @classmethod
    def from_executor_and_loop(
        cls, executor: Executor, event_loop: asyncio.AbstractEventLoop, **kwargs
    ) -> "TopologicalTaskRunner":
        """Create a TopologicalTaskRunner with a custom executor and event loop.
        Args:
            executor (Executor): The custom executor to use.
            loop (asyncio.AbstractEventLoop): The custom event loop to use.
        """
        return cls(executor=executor, event_loop=event_loop, **kwargs)

    @property
    def failed_tasks(self) -> List[TPTask]:
        return list(self._failed_tasks.values())

    def _set_results(
        self,
        ready_names: tuple[str, ...],
        ready_task_results: list[Any | BaseException],
        task_map: Dict[str, TPTask],
    ):
        for name, result in zip(ready_names, ready_task_results):
            task = task_map[name]
            if isinstance(result, BaseException):
                self.logger.error(f"Error executing task {name}: {result}")
                self._failed_tasks[name] = task
                self.task_errors[name] = result
            else:
                self.task_results[name] = result
                self._failed_tasks.pop(name, None)
                self.task_errors.pop(name, None)

    async def _execute_task(
        self, name: str, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        start_time = time()
        self.logger.info(f"Executing {name}")
        result = await (
            func(*args, **kwargs)
            if asyncio.iscoroutinefunction(func)
            else asyncio.to_thread(func, *args, **kwargs)
        )
        elapsed_time = time() - start_time
        elapsed_time_ms = elapsed_time * 1000
        self.logger.info(
            f"Executed {name} in {elapsed_time:.2f}s ({elapsed_time_ms:.0f}ms)"
        )
        return result

    def _calculate_ready_tasks(
        self,
        ready_names: tuple[str, ...],
        task_func_map: Dict[str, Callable[..., Any]],
        task_deps_map: Dict[str, List[str]],
        task_map: Dict[str, TPTask],
    ) -> List[asyncio.Future]:
        ready_tasks: List[asyncio.Future] = []

        for name in ready_names:
            func = task_func_map[name]
            task_deps = task_deps_map.get(name)

            dep_results = {dep: self.task_results[dep] for dep in task_deps}

            args = task_map[name].args
            kwargs = {**dep_results, **task_map[name].kwargs}

            ready_tasks.append(
                asyncio.ensure_future(self._execute_task(name, func, *args, **kwargs))
            )

        return ready_tasks

    def _map_tasks_functions_and_dependencies(
        self, tasks_to_run: List[TPTask]
    ) -> Tuple[Dict[str, TPTask], Dict[str, Callable[..., Any]], Dict[str, List[str]]]:
        task_map: Dict[str, TPTask] = {}
        task_func_map: Dict[str, Callable[..., Any]] = {}
        task_deps_map: Dict[str, List[str]] = {}

        for task in tasks_to_run:
            func = task.func

            name = task.name or func.__name__
            if name in task_map:
                raise ValueError(f"Duplicate task name detected: {name}")

            task_map[name] = task
            task_func_map[name] = func
            task_deps_map[name] = task.deps

        return task_map, task_func_map, task_deps_map

    def format_results(self, task_map: Dict[str, TPTask]) -> Dict[str, Any]:
        """
        Extract the final results dict from the task results based on result keys.
        """
        formatted_results: Dict[str, Any] = {}

        for name, result in self.task_results.items():
            task = task_map[name]
            result_key = task.res_key
            result_keys = result_key.split(".") if result_key else None
            if result_keys is not None:
                cur_results = formatted_results
                for key in result_keys[:-1]:
                    cur_results = cur_results.setdefault(key, {})
                cur_results[result_keys[-1]] = result

        return formatted_results

    async def arun(self, tasks_to_run: List[TPTask]) -> Dict[str, Any]:
        """Execute tasks in topological order and handle dependencies."""
        start_time = time()
        task_map, task_func_map, task_deps_map = (
            self._map_tasks_functions_and_dependencies(tasks_to_run)
        )

        ts = TopologicalSorter(task_deps_map)
        ts.prepare()

        while ts.is_active():
            ready_names = ts.get_ready()

            ready_tasks = self._calculate_ready_tasks(
                ready_names, task_func_map, task_deps_map, task_map
            )

            ready_task_results = await asyncio.gather(
                *ready_tasks, return_exceptions=True
            )

            self._set_results(ready_names, ready_task_results, task_map)

            ts.done(*ready_names)

        elapsed_time = time() - start_time
        elapsed_time_ms = elapsed_time * 1000
        self.logger.info(
            f"Executed tasks in {elapsed_time:.2f}s ({elapsed_time_ms:.0f}ms)"
        )

        return self.format_results(task_map)

    def run(self, tasks_to_run: List[TPTask]) -> Dict[str, Any]:
        """Main entry point for executing tasks."""
        coro = self.arun(tasks_to_run)
        try:
            return (
                asyncio.ensure_future(coro)
                if self.loop.is_running()
                else self.loop.run_until_complete(coro)
            )
        except RuntimeError:
            return asyncio.run(coro)

    def reset(self):
        self._failed_tasks.clear()
        self.task_results.clear()
        self.task_errors.clear()
        self.executor.shutdown(wait=False)


if __name__ == "__main__":
    runner = TopologicalTaskRunner()

    async def task1(arg: str, *args, **kwargs) -> Dict[str, Any]:
        await asyncio.sleep(1)
        return {"arg": arg, "task1": "Result 1"}

    async def task2(arg: str, task1: str, *args, **kwargs) -> Dict[str, Any]:
        await asyncio.sleep(1)
        return {"arg": arg, "task1": task1, "task2": "Result 2"}

    def task3(task1: str, *args, **kwargs) -> Dict[str, Any]:
        sleep(1)
        return {"task1": task1, "task3": "Result 3"}

    def task4(task2: str, task3: str, *args, **kwargs) -> Dict[str, Any]:
        sleep(1)
        return {"task2": task2, "task3": task3, "task4": "Result 4"}

    async def task5(task4: str, *args, **kwargs) -> Dict[str, Any]:
        await asyncio.sleep(1)
        return {"task4": task4, "task5": "Result 5"}

    arg = "Extra arg"

    task_results = runner.run(
        [
            TPTask(func=task1, deps=[], args=[arg]),
            TPTask(func=task2, args=[arg], deps=["task1"]),
            TPTask(func=task3, deps=["task1"]),
            TPTask(func=task4, deps=["task2", "task3"]),
            TPTask(func=task5, deps=["task4"]),
        ]
    )

    total_retries = 3
    for retries in range(total_retries):
        failed_tasks = runner.failed_tasks
        if not failed_tasks:
            break

        logging.info(
            f"Re-trying failed tasks {retries}/{total_retries}: {failed_tasks}"
        )
        task_results = runner.run(failed_tasks)

    formatted_results: Dict[str, Any] = {}
    for result in task_results.values():
        formatted_results.update(result)

    logging.info(f"Results: {formatted_results}")
    logging.info("All tasks executed successfully")
