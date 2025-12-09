# 目录
- [目录](#目录)
- [GEHRL 模型介绍](#gehrl-模型介绍)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本与示例代码](#脚本与示例代码)
    - [脚本参数](#脚本参数)
- [随机性说明](#随机性说明)
- [模型库主页](#模型库主页)

# [GEHRL 模型介绍](#目录)
本文提出了一种面向**目标导向学习路径推荐**的新型**图增强分层强化学习（Graph Enhanced Hierarchical Reinforcement Learning, GEHRL）**框架。该框架将学习路径推荐任务拆解为两个核心模块：**子目标选择（规划阶段）**与**子目标达成（学习项目推荐阶段）**。

具体而言，框架引入一个**高层智能体**作为子目标选择器，为低层智能体指定需要完成的子目标；而**低层智能体**的核心任务是为学习者推荐适配的学习项目。
为了保证推荐路径仅包含与目标相关的学习项目，从而提升目标达成效率，我们基于子目标与知识图谱设计了一个**基于图结构的候选筛选器**，以此约束低层智能体的动作空间。同时，我们为低层智能体的训练过程设计了**基于测试的内部奖励机制**，有效缓解了外部奖励稀疏性问题。

本研究成果以 *Graph Enhanced Hierarchical Reinforcement Learning for Goal-oriented Learning Path Recommendation* 为题，发表于 **CIKM 2023** 国际会议。

# [数据集](#目录)
- [assist15 数据集](https://sites.google.com/site/assistmentsdata/home/2015-assistments-skill-builder-data)

# [模拟器](#目录)
- KESassist15

# [环境要求](#目录)
- **硬件**：CPU 与 GPU
    - 需配备 CPU 处理器和英伟达（Nvidia）GPU 的硬件环境
- **框架**
    - [MindSpore-2.0.0](https://www.mindspore.cn/install/zh-CN)
- **依赖库**
  - numpy
  - tqdm
  - longling
  - mindspore==2.0.0
  - gym==0.22.0
  - scikit-learn
  - genism

- 更多详情请参考以下资源：
  - [MindSpore 教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
  - [MindSpore Python API 文档](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.html)

# [快速开始](#目录)
通过官网完成 MindSpore 安装后，可按照以下步骤启动训练与评估：

- **GPU 环境运行**
  ```shell
  # 第一步：构建模拟器
  python scripts/dataProcess.py
  # 第二步：启动训练与评估
  python scripts/runSim.py -s simulator -a model
  ```

# [脚本说明](#目录)
## [脚本与示例代码](#目录)
```text 
.
|-GEHRL
  |-README.md             # GEHRL 项目说明文档
  |-EduSim                # 模拟器与智能体核心代码目录
    |-__init__.py
    |-AbstractAgent.py    # 智能体抽象基类
    |-buffer.py           # 经验回放缓冲区
    |-deep_model.py       # 基础深度学习模型（如 GRU、GCN 等）
    |-GraphEmbedding.py   # 基于 Node2Vec 的图嵌入模型
    |-agents              # 智能体实现代码目录
      |-__init__.py
      |-AC.py             # 演员-评论家（Actor-Critic）智能体
      |-PPO.py            # 近端策略优化（PPO）智能体
      |-HRL.py            # 分层强化学习（HRL）智能体
    |-Envs                # 模拟器环境代码目录
      |-__init__.py
      |-KES
      |-KES_ASSIST15
      |-meta
      |-shared
    |-SimOs               # 模拟器系统层代码目录
      |-__init__.py
      |-SimOs.py          # 模拟器系统核心逻辑
    |-spaces              # 动作空间与观测空间定义代码
    |-utils               # 工具函数库
  |-scripts               # 数据处理与模型训练脚本目录
    |-runSim.py           # 训练与评估流程入口脚本
    |-dataProcess.py      # 数据预处理流程入口脚本
```

## [脚本参数](#目录)
- **runSim.py 脚本参数**
  详细参数定义及说明请参考 [scripts/runSim.py](./scripts/runSim.py)

[//]: # (# [模型说明]&#40;#目录&#41;)

[//]: # ()
[//]: # (## [性能指标]&#40;#目录&#41;)

[//]: # ()
[//]: # (### 训练性能)

[//]: # ()
[//]: # (| 参数                | GPU 环境配置                                                                                                                |)

[//]: # (|---------------------|----------------------------------------------------------------------------------------------------------------------------|)

[//]: # (| 硬件资源            | AMD Ryzen 2990WX 32 核处理器；256G 内存；NVIDIA GeForce 2080Ti 显卡                                                       |)

[//]: # (| 上传日期            | 2023年12月31日                                                                                                             |)

[//]: # (| MindSpore 版本      | 2.0.0                                                                                                                      |)

[//]: # (| 数据集              | assist15                                                                                                                   |)

[//]: # (| 模拟器              | KESassist15                                                                                                                |)

[//]: # (| 训练参数            | 最大步数 max_steps=20，最大回合数 max_episode_num=15000，学习率 lr=1e-5                                                    |)

[//]: # (| 优化器              | Adam                                                                                                                       |)

[//]: # (| 损失函数            | 策略梯度（Policy Gradient）                                                                                                |)

[//]: # (| 输出结果            | 奖励值（Reward）                                                                                                           |)

[//]: # (| 训练结果            | 基于模拟器的训练过程存在较强随机性，不同基线模型的性能对比请参考原论文                                                   |)

[//]: # (| 单步耗时            | 54.97 毫秒                                                                                                                 |)

[//]: # ()
[//]: # (### 推理性能)

[//]: # ()
[//]: # (| 参数                | GPU 环境配置                                                                                                                |)

[//]: # (|---------------------|-----------------------------------------------------------------------------------------------------------------------------|)

[//]: # (| 硬件资源            | AMD Ryzen 2990WX 32 核处理器；256G 内存；NVIDIA GeForce 2080Ti 显卡                                                        |)

[//]: # (| 上传日期            | 2023年01月15日                                                                                                              |)

[//]: # (| MindSpore 版本      | 1.9.10                                                                                                                      |)

[//]: # (| 数据集              | assist09、junyi                                                                                                             |)

[//]: # (| 模拟器              | DKT、CoKT                                                                                                                   |)

[//]: # (| 输出结果            | 奖励值（Reward）                                                                                                            |)

[//]: # (| 推理结果            | 基于模拟器的推理过程存在较强随机性，不同基线模型的性能对比请参考原论文                                                      |)

[//]: # (| 单步耗时            | 40.61 毫秒                                                                                                                  |)

# [随机性说明](#目录)
本项目的随机性主要来源于两个方面：
1.  模拟器训练过程的固有随机性
2.  模型权重的随机初始化

# [模型库主页](#目录)
请参考官方 [MindSpore 模型库](https://gitee.com/mindspore/models)

---
