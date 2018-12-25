import numpy as np

from entity.workflow import Workflow

# TODO(hang): get data from .csv file
time_reward_matrix = [[0.1270, 0.1746, 0.8191, 1.6822, 3.2659],
                      [0.1643, 0.2236, 1.0270, 2.0975, 4.0479],
                      [0.1640, 0.2235, 1.0255, 2.0948, 4.0519],
                      [0.1656, 0.2251, 1.0312, 2.1049, 4.0768],
                      [0.2008, 0.2737, 1.2588, 2.5725, 4.9698],
                      [0.2005, 0.2729, 1.2553, 2.5651, 4.9564],
                      [0.2011, 0.2743, 1.2619, 2.5784, 4.9790]]

cost_reward_matrix = [[0.021844, 0.0300312, 0.1408852, 0.2893384, 0.5617348],
                      [0.0157728, 0.0214656, 0.098592, 0.20136, 0.3885984],
                      [0.062976, 0.085824, 0.393792, 0.8044032, 1.5559296],
                      [0.0553104, 0.0751834, 0.3444208, 0.7030366, 1.3616512],
                      [0.03726848, 0.05079872, 0.23363328, 0.477456, 0.92239488],
                      [0.02005, 0.02729, 0.12553, 0.25651, 0.49564],
                      [0.07464832, 0.10182016, 0.46841728, 0.95710208, 1.8482048]]


class Env:
    def __init__(self, n_vm, n_agent):
        self.n_vm = n_vm
        self.n_actions = self.n_vm
        self.n_features = 1 + 7  # task_type and vm_state
        self.n_task = 138
        self.dim_state = self.n_task
        self.n_agent = n_agent
        # self.time_reward_matrix = np.random.rand(self.n_vm, 4)

        self.workflow = None
        self.task = None
        self.vm_time = None
        self.vm_cost = None
        self.released = None
        self.start_time = None
        self.task_exec = None
        self.state = None
        self.done = None
        # self.reward = None
        self.reset()

    def reset(self):
        self.workflow = [Workflow(i) for i in range(5)]
        self.vm_time = np.zeros(self.n_vm)
        self.vm_cost = np.zeros(self.n_vm)
        self.released = [[], [], [], [], []]
        self.start_time = np.zeros(self.n_task)
        self.task_exec = []
        self.state = np.ones(self.n_task)
        base = 0
        for i in range(len(self.workflow)):
            if i != 0:
                base += self.workflow[i - 1].size
            for j in range(len(self.workflow[i].precursor)):
                # print(self.workflow[i].precursor[j])
                idle = base + self.workflow[i].precursor[j]
                self.state[idle] = 0
        # print(self.state)

        cnt = 0
        for i in range(self.dim_state):
            if self.state[i] == 0:
                cnt += 1
        if cnt == 1 or cnt == 0:
            index = 0
        else:
            index = np.random.randint(cnt - 1)
        for i in range(self.dim_state):
            if self.state[i] == 0 and index != 0:
                index -= 1
            elif self.state[i] == 0 and index == 0:
                self.task = i
                break

        self.done = False
        # self.reward = 0
        obs = []
        for i in range(self.n_agent):
            obs.append(self.observation(i))
        return obs

    def step(self, action):
        obs = []
        reward = []
        done = []

        self.set_action()
        # print('step')
        # reward.append(self.rewards(action, 0))
        # reward.append(reward[0])
        for i in range(self.n_agent):
            reward.append(self.rewards(action, i))
            obs.append(self.observation(i))
            done.append(self.is_done())

        return obs, reward, done

    @staticmethod
    def has_value(arry, value):
        for i in range(len(arry)):
            if arry[i] == value:
                return True
        return False

    def release_node(self, task):
        # print(col)

        release = []
        count = 0
        belong = []

        for i in range(len(self.workflow)):
            if task < count + self.workflow[i].size:
                belong.append(i)
                belong.append(task - count)
                break
            count += self.workflow[i].size

        # print(belong)
        # print(self.scenario.workflows[belong[0]].structure)
        # print(self.scenario.node)
        self.released[belong[0]].append(belong[1])
        # print(self.scenario.node)

        back_node = []
        for i in range(self.workflow[belong[0]].size):
            if self.workflow[belong[0]].structure[belong[1]][i] == 1:
                back_node.append(i)

        # print(back_node)

        for i in range(len(back_node)):
            for j in range(self.workflow[belong[0]].size):
                if self.workflow[belong[0]].structure[j][back_node[i]] == 1 and not self.has_value(
                        self.released[belong[0]], j):
                    break
                elif j == self.workflow[belong[0]].size - 1:
                    release.append([belong[0], back_node[i]])
        # print(release)
        return release

    def set_action(self):
        self.state[self.task] = 1

        release = self.release_node(self.task)

        # print(release)
        if len(release) != 0:
            # cnt = 0
            for i in range(len(release)):
                cnt = 0
                if release[i][0] != 0:
                    for j in range(release[i][0]):
                        cnt += self.workflow[j].size
                cnt += release[i][1]
                # for i in range(7):
                self.state[cnt] = 0

        # for i in range(len(self.dim_state)):
        #     if self.state[i] == 0:
        #         self.task = i
        #         break

    def observation(self, flag):
        # get task_type
        count = 0
        belong = []
        for i in range(len(self.workflow)):
            if self.task < count + self.workflow[i].size:
                belong.append(i)
                belong.append(self.task - count)
                break
            count += self.workflow[i].size
        # print(belong)
        task_type = self.workflow[belong[0]].subTask[belong[1]].task_type

        # TODO(hang): return all agent state, cus agents can observe each other
        if flag == 0:
            return np.concatenate(([task_type], self.vm_time), 0)
        else:
            return np.concatenate(([task_type], self.vm_cost), 0)

    def time_reward(self, action):
        # TODO(hang): design a smarter strategy
        strategy = []
        last_makespan = max(self.vm_time)

        count = 0
        belong = []
        for i in range(len(self.workflow)):
            if self.task < count + self.workflow[i].size:
                belong.append(i)
                belong.append(self.task - count)
                break
            count += self.workflow[i].size

        strategy.append(belong[0])
        strategy.append(action + 1)

        task_type = self.workflow[belong[0]].subTask[belong[1]].task_type

        # print(agent.state.pos, type)
        exec_time = time_reward_matrix[action][task_type]
        if self.vm_time[action] >= self.start_time[self.task]:
            strategy.append(self.vm_time[action])
            self.vm_time[action] += exec_time
            strategy.append(self.vm_time[action])
        else:
            strategy.append(self.start_time[self.task])
            self.vm_time[action] = self.start_time[self.task] + exec_time
            strategy.append(self.vm_time[action])
        # self.vm_time[action] += exec_time
        # print(vm_finish_time)
        # time = max(self.vm_time) - last_makespan

        # for i in range(self.dim_state):
        #     if self.state[i] == 0:
        #         self.task = i
        #         break

        self.task_exec.append(strategy)

        finish_time = self.vm_time[action]

        back_node = []
        for i in range(self.workflow[belong[0]].size):
            if self.workflow[belong[0]].structure[belong[1]][i] == 1:
                back_node.append(i)

        for i in range(len(back_node)):
            if finish_time > self.start_time[back_node[i]]:
                self.start_time[back_node[i]] = finish_time

        # return max(vm_finish_time)
        return last_makespan, exec_time
        # return pow((4.979 - reward) / 4.079, 2)

    def cost_reward(self, action):
        count = 0
        belong = []
        for i in range(len(self.workflow)):
            if self.task < count + self.workflow[i].size:
                belong.append(i)
                belong.append(self.task - count)
                break
            count += self.workflow[i].size
        task_type = self.workflow[belong[0]].subTask[belong[1]].task_type

        col = np.array(cost_reward_matrix)[:, task_type]
        worst = col[np.argmax(col)]
        best = col[np.argmin(col)]
        cost = cost_reward_matrix[action][task_type]
        self.vm_cost[action] += cost

        cnt = 0
        for i in range(self.dim_state):
            if self.state[i] == 0:
                cnt += 1
        if cnt == 1 or cnt == 0:
            index = 0
        else:
            index = np.random.randint(cnt - 1)
        for i in range(self.dim_state):
            if self.state[i] == 0 and index != 0:
                index -= 1
            elif self.state[i] == 0 and index == 0:
                self.task = i
                break

        return best, worst, cost

    def rewards(self, action, flag):
        # last_makespan, exec_time = self.time_reward(action)
        # inc_makespan = max(self.vm_time) - last_makespan
        # b_cost, w_cost, a_cost = self.cost_reward(action)
        # return (pow((exec_time - inc_makespan) / exec_time, 3) + pow((w_cost - a_cost) / (w_cost - b_cost), 3)) / 2
        if flag == 0:
            last_makespan, exec_time = self.time_reward(action)
            inc_makespan = max(self.vm_time) - last_makespan
            inc_makespan = round(inc_makespan, 4)

            # if inc_makespan == exec_time:
            #     print('time punish')
            #     return -0.5
            # else:
            # print(np.var(self.vm_time))
            return pow((exec_time - inc_makespan) / exec_time, 3)
        else:
            b_cost, w_cost, a_cost = self.cost_reward(action)

            # if w_cost == a_cost:
            #     print('cost punish')
            #     return -0.5
            # else:
            # print(pow((w_cost - a_cost) / (w_cost - b_cost), 3))
            return pow((w_cost - a_cost) / (w_cost - b_cost), 3)

    def is_done(self):
        for i in self.state:
            if i != 1:
                return False
        return True
