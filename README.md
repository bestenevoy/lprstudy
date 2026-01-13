export GEHRL_DEVICE_TARGET=GPU

export  CUDA_VISIBLE_DEVICES=0
uv run runSim.py -s KESassist15 -e 1 -a HRL -r 1 -m 20

export  CUDA_VISIBLE_DEVICES=1
uv run runSim.py -s KESassist15 -e 1 -a HRL -r 2 -m 20

export  CUDA_VISIBLE_DEVICES=2
uv run runSim.py -s KESassist15 -e 1 -a HRL -r 3 -m 20

**参数说明**

EduSim/Experiment_logs/KESassist15/Experiment_2/1.txt

EduSim/Experiment_logs/KESassist15/Experiment_<-e>/<-r>.txt

-s/--simulator：模拟器环境，目前只有 KESassist15
-a/--agent：智能体类型，目前只有 HRL
-e/--experiment_idx：实验编号，用于日志目录（Experiment_<id>）
-r/--repeat_num：随机种子编号（1/2/3 → seeds=[1,5,10]）
-m/--max_steps：每个 episode 的最大步数（对应论文 Step=5/10/20）
--max_episode_num：总 episode 数（默认 15000）



export  CUDA_VISIBLE_DEVICES=3


## 执行步骤

下面是按顺序整理的命令清单（Linux/Bash 版），你可以直接在 Linux 上重跑一遍。

0) 放数据

# 原始文件放这里
data/ASSISTments2015/2015_100_skill_builders_main_problems.csv
1) 生成 processed 数据

python scripts/assist15_process.py
2) 生成知识图

mkdir -p data/dataProcess/ASSISTments2015
python -m EduSim.Envs.KES_ASSIST15.BuildGraph
3) 生成图嵌入

mkdir -p EduSim/Envs/meta_data
python -m EduSim.GraphEmbedding
4) 训练 DKT（环境）

export GEHRL_DEVICE_TARGET=CPU
python -m EduSim.Envs.KES_ASSIST15.envDKT
5) 训练 DKT（智能体）

export GEHRL_DKT_TRAIN_GOAL=agent_DKT
python -m EduSim.Envs.KES_ASSIST15.envDKT
6) 运行主流程

python runSim.py -s KESassist15 -a HRL -r 1
如果你在 Linux 上有 GPU 版 MindSpore，可以把第 4、5、6 步的环境变量改成：

export GEHRL_DEVICE_TARGET=GPU