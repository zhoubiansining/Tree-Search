# Pysearch version 0.1

# 功能说明
Pysearch 提供包装好的 LLM 搜索框架及相应的算法，使用者可以使用提供的测试程序直接进行基于 CoT, ToT, MCTS 框架的数据集评测，也可以调用相关接口，
实现更丰富的测试与应用。使用前，请按照要求完成配置。

# 基本配置
1. 配置数据，运行测试之前，需要将测试数据集转换为标准化的 json 格式，每个 json 项包含键 "content" 及 "answer" 。以数据集名称为文件夹名，其中包含若干 json 文件，
即不同的测试数据，需放置在 "data" 路径下。


2. 配置模型，如果采用 gpt 或 glm 的 api 服务，则需要在 "models/model.py" 文件中的 api settings 相应位置配置 api-key，api-base 与 url 等。
如果采用本地的推理/价值模型，需要在 "models/model.py" 处设定模型 checkpoint 路径。

# 参数解释
1.task_name: 任务/数据集名称，如scibench。相应的测试数据集应放置在"data"路径下的同名文件夹内。 

2.file: 数据集 json 文件名，可针对具体类别/学科的数据进行测试。

3.propose_method: 推理模型，可选择 glm， gpt 以及本地模型 local。

4.value_method: 价值模型，可选择 glm， gpt 以及本地模型 local。

5.mode: 推理框架，支持 CoT，ToT 以及 MCTS。

6.temperature: 搜索温度，用于决定生成回复的自由度。

7.time_limit: MCTS 框架中设定的搜索时间上限。

8.iteration_limit: MCTS 框架中设定的搜索轮次上限。

9.roll_policy: MCTS 框架中蒙特卡洛模拟的策略，可选随机策略 random 或贪心策略 greedy。

10.exploration_constant: MCTS 框架中筛选节点时较少访问次数的重要程度，搜索更看重已有价值还是访问次数。

11.roll_forward_steps: MCTS 框架中蒙特卡洛模拟的前进步数。

12.end_gate: ToT 和 MCTS 框架中判定结束的最低价值阈值。

13.branch: ToT 和 MCTS 框架中节点扩展的下一步分支数。

14.roll_branch: MCTS 框架中蒙特卡洛模拟的采样分支数。

15.inf: MCTS 框架中未访问节点的基本价值。

16.evaluate: 是否进行结果评测。

17.alpha: MCTS 框架中蒙特卡洛模拟的价值更新权重。

18.visualize: ToT 和 MCTS 框架中结果是否进行树状图可视化。

19.use_case_prompt: 是否采用样例输出 prompt 辅助生成。

20.use_reflection: 是否采用反思意见机制，目前仅支持 MCTS 框架。

21.low: 节点价值的下界。

22.high: 节点价值的上界。

23.algorithm: ToT 框架中采用的算法，可选择深度优先 dfs 或者广度优先 bfs。

24.select_branch: ToT 框架中子节点中实际选择下探的节点数。

25.max_depth: ToT 框架中允许的最大下探步数（深度）。

26.select_method: ToT 框架中选择下探节点的方法，可选择贪心算法 greedy 或者采样方法 sample。
