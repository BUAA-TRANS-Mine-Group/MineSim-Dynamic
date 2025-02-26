import copy

import numpy as np
from tqdm import tqdm


from devkit.common.coordinate_system.frenet import JerkSpaceSamplingFrenetTrajectory


class PolyVTSampling(object):
    """基于线型设计的采样算法"""

    def __init__(
        self, total_time: float, delta_t: float, jerk_min: float, jerk_max: float, a_min: float, a_max: float, v_max: float, num_samples: float
    ) -> None:
        """参数主要是自车的动力学最大性能限制."""
        self.total_time = total_time
        self.delta_t = delta_t
        self.jerk_min = jerk_min
        self.jerk_max = jerk_max
        self.a_min = a_min
        self.a_max = a_max
        self.v_max = v_max
        self.num_samples = num_samples
        self.time_step_max = int(round(self.total_time / self.delta_t))
        self.time_array = np.linspace(0.0, total_time, int(total_time / delta_t) + 1).tolist()
        # 为每个时间点生成jerk值
        self.jerk_values = np.linspace(jerk_min, jerk_max, num_samples)
        self.t1_array_delta = 0.4000
        self.t2_array_delta = 0.8000
        self.t3_array_delta = 1.0
        self.t4_array_delta = 2.0
        self.t1_array = np.arange(0.0, self.total_time + 1e-9, self.t1_array_delta)
        # self.t2_array = np.concatenate((np.array([5.0]), np.arange(0.0, self.total_time + 1e-9, self.t2_array_delta)))
        self.t2_array = np.arange(0.0, self.total_time + 1e-9, self.t2_array_delta)
        self.t3_array = np.arange(0.0, self.total_time + 1e-9, self.t3_array_delta)
        self.t4_array = np.arange(0.0, self.total_time + 1e-9, self.t4_array_delta)

        self.jerk_min_3 = jerk_min * 0.51
        self.jerk_max_3 = jerk_max * 0.51
        self.jerk_min_5 = jerk_min * 0.21
        self.jerk_max_5 = jerk_max * 0.21

    @staticmethod
    def calculate_st_by_jt(s0: float, v0: float, a0: float, j: float, t: float):
        """计算jt采样结果对应时刻的位移s"""
        return s0 + v0 * t + 0.5 * a0 * t * t + (1 / 6.0) * j * t * t * t

    @staticmethod
    def calculate_vt_by_jt(v0: float, a0: float, j: float, t: float):
        """计算jt采样结果对应时刻的速度v"""
        return v0 + a0 * t + 0.5 * j * t * t

    @staticmethod
    def calculate_at_by_jt(a0: float, j: float, t: float):
        """计算jt采样结果对应时刻的加速度a"""
        return a0 + j * t

    def check_state(self, st: float, vt: float, at: float, s0: float, v0: float, a0: float):
        """检查终端状态是否超过动力学约束

        Args:
        s0, v0, a0:初始状态值;
        st, vt, at:t时间后的状态值;


        Returns:
            bool: _description_
        """

        # check acceleration
        ACC_MAX = self.a_max
        if at > ACC_MAX or at < -ACC_MAX:
            return False

        # check velocity
        VEL_MAX = self.v_max
        if vt > VEL_MAX or vt < 0.0:
            return False

        if st < s0:
            return False

        return True

    def sampling(self, s0: float, v0: float, a0: float) -> list[tuple]:
        """基于a-t空间线形设计的j-t空间采样策略.
        短规划时域(如5秒):短规划时域内,好的纵向规划是不进行频繁的加速、减速模式切换的;
        设计5段at直线拼接,对应着不同的驾驶模式:
        1)a曲线上升;2)a曲线恒定不变;3)a曲线下降\反方向上升; 4)a曲线再次恒定不变;5)a曲线反方向下降;
        对应的每段a-t直线的jerk值及维持时间:j1, t1, j2, t2, j3, t3, j4, t4, j5, t5

        Args:
            s0, v0, a0: Initial value.

        Returns:
            list[tuple]: 由(j1, t1, j2, t2, j3, t3, j4, t4, j5, t5)组成的不同at线形直线拼接采样结果种子.
        """
        seeds = []
        TINTER_MIN = 0.65
        for j1 in tqdm(np.linspace(self.jerk_min, self.jerk_max, self.num_samples)):
            for t1 in self.time_array:
                # j1固定数量采样,t1固定数量采样
                if t1 < TINTER_MIN:
                    continue

                s1 = self.calculate_st_by_jt(s0, v0, a0, j1, t1)
                v1 = self.calculate_vt_by_jt(v0, a0, j1, t1)
                a1 = self.calculate_at_by_jt(a0, j1, t1)

                if not self.check_state(s1, v1, a1, s0, v0, a0):
                    continue

                j2 = 0
                # for t2 in np.linspace(0, self.total_time - t1, int((self.total_time - t1) / (self.delta_t * 2))):
                for t2 in np.linspace(0, self.total_time - t1, int((self.total_time - t1) / (self.delta_t * 4))):
                    # j2=0,t2采样数量是t1计算方式/2
                    if t1 + t2 >= self.total_time - self.delta_t * 3:
                        continue

                    s2 = self.calculate_st_by_jt(s1, v1, a1, j2, t2)
                    v2 = self.calculate_vt_by_jt(v1, a1, j2, t2)
                    a2 = self.calculate_at_by_jt(a1, j2, t2)

                    if not self.check_state(s2, v2, a2, s1, v1, a1):
                        continue

                    if t1 + t2 >= self.total_time:
                        continue

                    for j3 in np.linspace(self.jerk_min, self.jerk_max, int(self.num_samples / 4)):
                        if j3 * j1 > 0:  # 线型设计，不允许只有同向加速度变化
                            continue

                        for t3 in self.t3_array:
                            if t1 + t2 + t3 >= self.total_time - self.delta_t * 2:
                                continue

                            s3 = self.calculate_st_by_jt(s2, v2, a2, j3, t3)
                            v3 = self.calculate_vt_by_jt(v2, a2, j3, t3)
                            a3 = self.calculate_at_by_jt(a2, j3, t3)

                            if not self.check_state(s3, v3, a3, s2, v2, a2):
                                continue

                            if a3 * a1 > 0:  # 确保第三段模式已经变为相反的纵向运动模型
                                continue

                            if a3 * a2 < 0:  #! 检查一下[a3 ,0.0 ,a2]的极点情况
                                t2_5 = (0.0 - a2) / j3
                                v2_5 = self.calculate_vt_by_jt(v2, a2, j3, t2_5)
                                if v2_5 > self.v_max or v2_5 < 0.0 + 1e-9:
                                    continue

                            j4 = 0
                            for t4 in self.t4_array:
                                # j4=0,t4采样数量是t1计算方式/8
                                if t1 + t2 + t3 + t4 >= self.total_time - self.delta_t:
                                    continue

                                s4 = self.calculate_st_by_jt(s3, v3, a3, j4, t4)
                                v4 = self.calculate_vt_by_jt(v3, a3, j4, t4)
                                a4 = self.calculate_at_by_jt(a3, j4, t4)

                                if not self.check_state(s4, v4, a4, s3, v3, a3):
                                    continue

                                # j5=保证加速度到0
                                t5 = self.total_time - t1 - t2 - t3 - t4
                                j5 = (0.0 - a4) / t5
                                if j5 > self.jerk_max or j5 < self.jerk_min:
                                    continue

                                s5 = self.calculate_st_by_jt(s4, v4, a4, j5, t5)
                                v5 = self.calculate_vt_by_jt(v4, a4, j5, t5)
                                a5 = self.calculate_at_by_jt(a4, j5, t5)
                                if not self.check_state(s5, v5, a5, s4, v4, a4):
                                    continue

                                seeds.append((j1, t1, j2, t2, j3, t3, j4, t4, j5, t5))
        return seeds

    def get_all_frenet_trajectory(self, s0: float, a0: float, v0: float, seeds: float, ft: JerkSpaceSamplingFrenetTrajectory):
        """批量生成速度轨迹"""
        return [self.get_one_frenet_trajectory(s0=s0, v0=v0, a0=a0, seed=seed, jssft=ft) for seed in seeds]

    def get_one_frenet_trajectory(
        self, s0: float, v0: float, a0: float, seed: float, jssft: JerkSpaceSamplingFrenetTrajectory
    ) -> JerkSpaceSamplingFrenetTrajectory:
        """根据jerk-t空间采样结果生成一条速度轨迹"""
        ft = copy.deepcopy(jssft)
        j1, t1, j2, t2, j3, t3, j4, t4, j5, t5 = seed

        s1 = self.calculate_st_by_jt(s0, v0, a0, j1, t1)
        v1 = self.calculate_vt_by_jt(v0, a0, j1, t1)
        a1 = self.calculate_at_by_jt(a0, j1, t1)

        s2 = self.calculate_st_by_jt(s1, v1, a1, j2, t2)
        v2 = self.calculate_vt_by_jt(v1, a1, j2, t2)
        a2 = self.calculate_at_by_jt(a1, j2, t2)

        s3 = self.calculate_st_by_jt(s2, v2, a2, j3, t3)
        v3 = self.calculate_vt_by_jt(v2, a2, j3, t3)
        a3 = self.calculate_at_by_jt(a2, j3, t3)

        s4 = self.calculate_st_by_jt(s3, v3, a3, j4, t4)
        v4 = self.calculate_vt_by_jt(v3, a3, j4, t4)
        a4 = self.calculate_at_by_jt(a3, j4, t4)

        for t in jssft.t:
            dt = 0
            if t < t1:
                dt = t
                ft.s_ddd.append(j1)
                ft.s_dd.append(self.calculate_at_by_jt(a0, j1, dt))
                ft.s_d.append(self.calculate_vt_by_jt(v0, a0, j1, dt))
                ft.s.append(self.calculate_st_by_jt(s0, v0, a0, j1, dt))

            elif t < t1 + t2:
                dt = t - t1
                ft.s_ddd.append(j2)
                ft.s_dd.append(self.calculate_at_by_jt(a1, j2, dt))
                ft.s_d.append(self.calculate_vt_by_jt(v1, a1, j2, dt))
                ft.s.append(self.calculate_st_by_jt(s1, v1, a1, j2, dt))

            elif t < t1 + t2 + t3:
                dt = t - t1 - t2
                ft.s_ddd.append(j3)
                ft.s_dd.append(self.calculate_at_by_jt(a2, j3, dt))
                ft.s_d.append(self.calculate_vt_by_jt(v2, a2, j3, dt))
                ft.s.append(self.calculate_st_by_jt(s2, v2, a2, j3, dt))

            elif t < t1 + t2 + t3 + t4:
                dt = t - t1 - t2 - t3
                ft.s_ddd.append(j4)
                ft.s_dd.append(self.calculate_at_by_jt(a3, j4, dt))
                ft.s_d.append(self.calculate_vt_by_jt(v3, a3, j4, dt))
                ft.s.append(self.calculate_st_by_jt(s3, v3, a3, j4, dt))

            else:
                dt = t - t1 - t2 - t3 - t4
                ft.s_ddd.append(j5)
                ft.s_dd.append(self.calculate_at_by_jt(a4, j5, dt))
                ft.s_d.append(self.calculate_vt_by_jt(v4, a4, j5, dt))
                ft.s.append(self.calculate_st_by_jt(s4, v4, a4, j5, dt))

        return ft


if __name__ == "__main__":
    pass
