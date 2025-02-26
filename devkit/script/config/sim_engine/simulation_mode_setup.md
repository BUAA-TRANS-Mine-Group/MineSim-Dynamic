# simulation_mode setup read me

在 MineSim/devkit/script/run_simulation.py 中选择不同的全局配置 YAML 即可

## simulation_mode_0

| simulation mode No.  | 0                                                            |
| :------------------ | ------------------------------------------------------------ |
| Test Mode name      | Replay Test Mode(Ego open-loop with non-reactive agents)      |
| Applicable Scenario | Dynamic Scenario                                             |
| Explanation         | 1) Prediction Algorithms: Perfect Prediction;<br/>2) Ego Simulation: perfect_tracking_controller(open-loop);<br/>3) Agents Simulation: replay policy(replay_policy_agents_box_track). |



## simulation_mode_1

| simulation mode No.  | 1                                                            |
| :------------------ | ------------------------------------------------------------ |
| Test Mode name      | Replay Test Mode(Ego close-loop with non-reactive agents)      |
| Applicable Scenario | Dynamic Scenario                                             |
| Explanation         | 1) Prediction Algorithms: Perfect Prediction;<br/>2) Ego Closed-loop Simulation: LQR-based controller + KBM_wRL ego update model;<br/>3) Agents Simulation: replay policy(replay_policy_agents_box_track). |

## simulation_mode_2

| simulation mode No.  | 2                                                            |
| :------------------ | ------------------------------------------------------------ |
| Test Mode name      | Replay Test Mode(Ego close-loop withnon-reactive agents)      |
| Applicable Scenario | Dynamic Scenario                                             |
| Explanation         | 1) Prediction Algorithms: Perfect Prediction;<br/>2) Ego Closed-loop Simulation: LQR-based controller + KBM ego update model;<br/>3) Agents Simulation: replay policy(replay_policy_agents_box_track). |

## simulation_mode_3

| simulation mode No.  | 3                                                            |
| :------------------ | ------------------------------------------------------------ |
| Test Mode name      | Replay Test Mode(Ego close-loop withnon-reactive agents)      |
| Applicable Scenario | Dynamic Scenario                                             |
| Explanation         | 1) Prediction Algorithms: Perfect Prediction;<br/>2) Ego Closed-loop Simulation: iLQR-based controller + KBM_wRL ego update model;<br/>3) Agents Simulation: replay policy(replay_policy_agents_box_track). |


## simulation_mode_4

| simulation mode No.  | 4                                                            |
| :------------------ | ------------------------------------------------------------ |
| Test Mode name      | Interactive Test Mode (Ego closed-loop with reactive agents)      |
| Applicable Scenario | Dynamic Scenario                                             |
| Explanation         | 1) Prediction Algorithms: Perfect Prediction;<br/>2) Ego Closed-loop Simulation: LQR-based controller + KBM_wRL ego update model;<br/>3) Agents Simulation:  reactive agents(reactive_policy_agents_idm). |

