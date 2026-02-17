根据提供的论文《Safe LoRA: the Silver Lining of Reducing Safety Risks when Fine-tuning Large Language Models》 以及代码文件信息，以下是复现 Safe LoRA 实验效果的详细流程和方法说明。

Safe LoRA 的核心理念是**“训练后处理”（Post-hoc）**，即先进行正常的 LoRA 微调，然后通过数学投影的方法修正权重，无需在微调过程中更改损失函数或增加安全数据 。

+2

### 一、 核心方法原理 (Methodology)

复现的核心在于构建“对齐矩阵”（Alignment Matrix）并利用它将微调后的权重投影回安全子空间。

1. **构建对齐矩阵 ($V$)**：
    
    - 利用一对“已对齐模型”（如 Llama-2-Chat）和“未对齐模型”（如 Llama-2-Base）的权重差异来定义安全方向 。
        
        +1
        
    - 公式：$V = W_{aligned} - W_{unaligned}$ 。
        
    - 在复现时，你可以直接使用官方发布的 Base 模型作为 $W_{unaligned}$，Chat/Instruct 模型作为 $W_{aligned}$ 。
        
2. **构建投影算子 ($C$)**：
    
    - 为了计算效率，论文推荐使用近似投影矩阵。
        
    - 公式：$C = \frac{VV^T}{\|V\|_F}$ 。
        
3. **安全投影 (Safety Projection)**：
    
    - 检查微调后的 LoRA 权重 ($\Delta W$) 与安全方向的相似度。如果相似度低于阈值 $\tau$，则强制投影 。
        
        +1
        
    - 修正公式：$\Delta W_{new} = C \cdot \Delta W$。
        

---

### 二、 详细复现步骤 (Step-by-Step Guide)

#### 第一步：准备模型与数据

1. **基础模型**：下载 Meta 官方的 Llama-2-7B-Base (作为未对齐参照) 和 Llama-2-7B-Chat (作为对齐参照及微调底座) 。
    
2. **数据集**：
    
    - **PureBad**：包含 100 条恶意指令，用于测试纯恶意微调场景 。
        
    - **Dialog Summary (SAMSum)**：选取 1000 条对话摘要数据，并混合 100 条 PureBad 恶意数据，模拟由良性与恶意数据混合的真实微调场景 。
        
    - **Alpaca**：作为纯良性数据集的对照组 。
        

#### 第二步：执行标准 LoRA 微调

使用标准的 LoRA 方法在上述数据集上微调 Llama-2-7B-Chat 模型。这一步**不需要**加入 Safe LoRA 的代码，就是常规微调 。

+1

- **目标模块**：仅针对 Attention 层的 `q_proj` 和 `v_proj` 。
    
- **LoRA 参数**：Rank ($r$) 设置为 8 。
    
- **超参数设置 (Llama-2-7B-Chat)** ：
    
    - Learning Rate: $5 \times 10^{-5}$
        
    - Batch Size: 5
        
    - Epochs: 5
        
    - Optimizer: 默认为 AdamW (根据 Meta 官方脚本推断)。
        

#### 第三步：执行 Safe LoRA 投影 (Post-hoc Patch)

这是复现的关键步骤，在微调完成后进行。你需要编写或运行脚本（如代码库中的 `model.py` 或 `SafeLoRA` 类）来加载微调好的 LoRA 权重并进行修改。

1. **加载权重**：加载 Base 模型权重、Chat 模型权重以及刚才微调好的 LoRA 权重 ($\Delta W = A B^T$)。
    
2. **计算层级相似度与投影**：
    
    对于每一层（Layer $i$）：
    
    - 计算对齐向量：$V^i = W_{chat}^i - W_{base}^i$。
        
    - 构建投影算子：$C^i = \frac{V^i {V^i}^T}{\|V^i\|_F}$。
        
    - 计算相似度分数：计算原始 LoRA 更新 ($\Delta W^i$) 与投影后更新 ($C^i \Delta W^i$) 之间的余弦相似度 。
        
        +1
        
3. **应用阈值 ($\tau$)**：
    
    - 设定阈值 $\tau$。例如在 Dialog Summary 实验中，阈值设为 **0.35** 。
        
    - 如果某层的相似度分数 $<\tau$，则用 $C^i \Delta W^i$ 替换该层的原始 LoRA 权重；否则保持不变 。
        
    - _注_：也可以选择投影相似度最低的 Top-K 层。在 Dialog Summary 实验中，大约有 7 层（约 11% 的层数）被投影 。
        
        +1
        

#### 第四步：评估 (Evaluation)

使用修正后的 LoRA 权重加载模型进行推理，并从以下三个维度评估：

1. **安全性 (Harmfulness Score)**：
    
    - 使用 GPT-4 对模型生成的回复打分（1-5分，1最安全，5最有害）。
        
    - 测试集：使用包含 11 个类别的恶意指令集（如非法活动、仇恨言论等）。
        
2. **可用性 (Utility)**：
    
    - 对于 Dialog Summary 任务，使用 **ROUGE-1 F1** 分数与 Ground Truth 进行比对 。
        
    - 对于 PureBad/Alpaca，使用 **MT-Bench** (GPT-4 打分 1-10) 。
        
3. **攻击成功率 (ASR)**：
    
    - 检查回复中是否**不包含**拒绝关键词（如 "I cannot", "I apologize" 等）。如果不包含拒绝词，视为攻击成功 。
        
        +1
        

---

### 三、 关键复现细节总结 (Cheatsheet)

为了确保复现结果与论文表 3 (Table 3) 一致，请严格遵守以下配置：

|**配置项**|**参数值 / 说明**|**来源**|
|---|---|---|
|**模型**|Llama-2-7B-Chat||
|**对齐矩阵来源**|Llama-2-7B-Chat (Aligned) - Llama-2-7B-Base (Unaligned)||
|**微调数据**|Dialog Summary (1000 samples) + 100 Harmful samples||
|**LoRA Rank**|8||
|**LoRA Modules**|`q_proj`, `v_proj`||
|**Learning Rate**|$5 \times 10^{-5}$||
|**相似度阈值**|0.35 (导致约 7 层被投影)||
|**投影操作**|Post-hoc (训练后一次性修补)||
|**预期结果**|ROUGE-1 F1 约 49.79%; Harmfulness Score 约 1.297||

### 四、 代码实现指引

根据你上传的文件列表，代码复现逻辑应位于 `ibm/safelora/SafeLoRA.../` 目录下：

1. **`model.py`**：应包含 `SafeLoRA` 类，负责加载 Base/Chat 模型，计算 $V$ 和 $C$，并执行权重修改逻辑。
    
2. **`SamSum.py`**：这应该是针对 Dialog Summary (SAMSum) 数据集的实验脚本。
    
3. **`config.py`**：可能包含超参数（如阈值 0.35、LoRA rank 8 等）。
    

你只需运行微调脚本得到 LoRA adapter，然后调用 `SafeLoRA` 类对该 adapter 进行处理即可。由于该方法是 "training-free"（无需额外训练过程，只需矩阵运算），处理速度非常快 。

+1