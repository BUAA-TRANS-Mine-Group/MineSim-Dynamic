import concurrent.futures
import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union

from devkit.sim_engine.callback.simulation_log_callback import SimulationLogCallback
from devkit.sim_engine.runner.abstract_runner import AbstractRunner
from devkit.sim_engine.runner.runner_report import RunnerReport
from devkit.sim_engine.runner.simulations_runner import SimulationRunner
from devkit.utils.multithreading.worker_pool import Task, WorkerPool

logger = logging.getLogger(__name__)


def run_simulation(sim_runner: AbstractRunner, exit_on_failure: bool = False) -> RunnerReport:
    """
    Proxy for calling simulation.
    :param sim_runner: A simulation runner which will execute all batched simulations.
    :param exit_on_failure: If true, raises an exception when the simulation fails.
    :return report for the simulation.
    """
    # Store start time so that if the simulations fail, we know how long they ran for
    #  记录当前时间，用于后续计算仿真耗时
    start_time = time.perf_counter()
    try:
        # !关键运行接口
        # 尝试运行仿真并返回运行报告
        return sim_runner.run()
    except Exception as e:
        # 如果过程中出现异常，提取错误堆栈信息
        error = traceback.format_exc()

        # 将错误信息打印到日志中  Print to the terminal
        logger.warning("#log#----------- Simulation failed: with the following trace:")
        traceback.print_exc()  # 打印详细的异常堆栈到控制台
        logger.warning(f"#log#Simulation failed with error:\n {e}")

        # 打印失败的场景日志名和场景名
        # Log the failed scenario log_name/scenario_name  sim_runner.scenario.log_name
        failed_scenarios = f"[{sim_runner.scenario.log_name}, {sim_runner.scenario.scenario_name}]\n"
        logger.warning(f"\n#log#Failed simulation [scenario log,scenario name]:\n {failed_scenarios}")
        logger.warning("#log#----------- Simulation failed!\n\n")

        # 如果配置了在失败时退出，则抛出 RuntimeError
        if exit_on_failure:
            raise RuntimeError("Simulation failed")

        # 记录结束时间
        end_time = time.perf_counter()
        # 构造一个 RunnerReport 对象，用于保存失败的运行信息
        report = RunnerReport(
            succeeded=False,  # 标记为失败
            error_message=error,  # 记录错误堆栈
            start_time=start_time,  # 仿真开始时间
            end_time=end_time,  # 仿真结束时间
            planner_report=None,  # 规划器报告，这里没有成功因此为 None
            scenario_name=sim_runner.scenario.scenario_name,  # 场景名称
            planner_name=sim_runner.planner.name(),  # 规划器名称
            log_name=sim_runner.scenario.log_name,  # 日志名
        )

        return report  # 返回构造好的失败报告


def execute_runners(
    runners: List[AbstractRunner],
    worker: WorkerPool,
    num_gpus: Optional[Union[int, float]],
    num_cpus: Optional[int],
    exit_on_failure: bool = False,
    verbose: bool = False,
) -> List[RunnerReport]:
    """
    Execute multiple simulation runners or metric runners.
    :param runners: A list of simulations to be run.
    :param worker: for submitting tasks.
    :param num_gpus: if None, no GPU will be used, otherwise number (also fractional) of GPU used per simulation.
    :param num_cpus: if None, all available CPU threads are used, otherwise number of threads used.
    :param exit_on_failure: If true, raises an exception when the simulation fails.
    """
    # Validating
    # [] [NOTE] 核心运行过程 # 断言必须至少要有一个 Runner，否则抛出异常
    assert len(runners) > 0, "No scenarios found to simulate!"

    # Start simulations  # 计算一共有多少个 Runner，需要执行多少次仿真
    number_of_sims = len(runners)
    # 输出到日志，告知用户将启动多少个仿真，以及使用何种 worker
    logger.info(f"#log# Starting {number_of_sims} simulations using {worker.__class__.__name__}!")

    # 调用 worker.map 提交任务：
    # - Task(fn=run_simulation, num_gpus=num_gpus, num_cpus=num_cpus) 表示执行 run_simulation 函数
    # - runners 为要处理的对象列表
    # - exit_on_failure: 如果为 True，遇到任务失败则退出
    # - verbose: 是否输出更多的调试信息
    reports: List[RunnerReport] = worker.map(
        Task(fn=run_simulation, num_gpus=num_gpus, num_cpus=num_cpus),
        runners,
        exit_on_failure,
        verbose=verbose,
    )

    # 将所有 report 结果存储到一个字典当中，以方便后续查询并更新错误信息
    # 字典 key 为 (scenario_name, planner_name, log_name)，value 为对应的 RunnerReport
    # Store the results in a dictionary so we can easily store error tracebacks in the next step, if needed
    results: Dict[Tuple[str, str, str], RunnerReport] = {(report.scenario_name, report.planner_name, report.log_name): report for report in reports}

    # Iterate over runners, finding the callbacks which may have run asynchronously, and gathering their results
    # 从传入的 runners 中筛选出 SimulationRunner 类型的 Runner
    simulations_runners = (runner for runner in runners if isinstance(runner, SimulationRunner))
    # 对每个 SimulationRunner，获取其对应的 simulation 对象和 runner 自身
    relevant_simulations = ((runner.simulation, runner) for runner in simulations_runners)

    # 找到所有可能异步执行的 callbacks（包括 MetricCallback 和 SimulationLogCallback），
    # 并获取它们产生的 future 列表，方便后续检查是否执行成功
    callback_futures_lists = (
        (callback.futures, simulation, runner)
        for (simulation, runner) in relevant_simulations
        for callback in simulation.callback.callbacks
        #  todo MetricCallback
        # if isinstance(callback, MetricCallback) or isinstance(callback, SimulationLogCallback)
        if isinstance(callback, SimulationLogCallback)
    )

    # 建立一个 future 到 (scenario_name, planner_name, log_name) 的映射
    # 这样我们可以根据 future 得到对应的 runner 报告，进而修改其错误信息等
    callback_futures_map = {
        future: (
            simulation.scenario.scenario_name,
            runner.planner.name(),
            simulation.scenario.log_name,
        )
        for (futures, simulation, runner) in callback_futures_lists
        for future in futures
    }

    # concurrent.futures.as_completed 可以迭代地返回已完成的 future
    for future in concurrent.futures.as_completed(callback_futures_map.keys()):
        try:
            future.result()  # 如果执行过程中抛出异常，会在这里触发
        except Exception:
            # 捕获回调执行过程中的任何异常，并记录到对应的 RunnerReport 中
            error_message = traceback.format_exc()
            runner_report = results[callback_futures_map[future]]
            runner_report.error_message = error_message
            runner_report.succeeded = False
            runner_report.end_time = time.perf_counter()

    # Notify user about the result of simulations
    # 仿真结束后，统计成功或失败的数量并打印失败的场景
    failed_simulations = str()  # 存储所有失败场景的信息
    number_of_successful = 0  # 用于计数成功的仿真
    runner_reports: List[RunnerReport] = list(results.values())  # 获取所有报告列表

    for result in runner_reports:
        if result.succeeded:  # 如果 succeeded 为 True 表示成功
            number_of_successful += 1
        else:  # 如果失败，打印警告日志，并记录到 failed_simulations
            logger.warning("#log# Failed Simulation.\n '%s'", result.error_message)
            failed_simulations += f"[{result.log_name}, {result.scenario_name}] \n"

    # 计算失败的数量
    number_of_failures = number_of_sims - number_of_successful
    # 输出成功、失败的统计
    logger.info(f"#log# Number of successful simulations: {number_of_successful}")
    logger.info(f"#log# Number of failed simulations: {number_of_failures}")

    # Print out all failed simulation unique identifier
    # 如果有失败，则打印所有失败的 [log, token]
    if number_of_failures > 0:
        logger.info(f"Failed simulations [log, token]:\n{failed_simulations}")

    # 返回所有 RunnerReport 列表
    return runner_reports
