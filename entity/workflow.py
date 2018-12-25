# import numpy as np

from data.XMLProcess import XMLtoDAG
from entity.subtask import SubTask

TYPE = ['./data/Sipht_29.xml', './data/Montage_25.xml', './data/Inspiral_30.xml', './data/Epigenomics_24.xml',
        './data/CyberShake_30.xml']
NUMBER = [29, 25, 30, 24, 30]
TASK_TYPE = [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2],
             [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4, 0, 1, 2, 3],
             [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 2],
             [0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 0, 1, 2],
             [0, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 2, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4]]


class Workflow:
    def __init__(self, num):
        self.id = num + 1

        self.type = TYPE[num]
        self.size = NUMBER[num]
        self.subTask = [SubTask((num + 1) * 1000 + i + 1, TASK_TYPE[num][i]) for i in range(self.size)]  # 子任务

        dag = XMLtoDAG(self.type, self.size)
        self.structure = dag.get_dag()  # 带权DAG
        self.precursor = dag.get_precursor()

        # print(self.precursor)
        # self.structure = np.delete(self.structure, self.precursor, 0)
        # self.structure = np.delete(self.structure, self.precursor, 1)
        # print(self.structure)
