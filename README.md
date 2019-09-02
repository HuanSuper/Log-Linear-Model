# Log-Linear-Model
<ol>
  <li>简介</li>使用Log Linear Model模型预测当前句子的词性序列
  <li><Linear Model模型/li>
    <ol>
      <li>特征提取</li>对数据进行常见的特征提取（0，1编码）
      <li>权重学习</li>与Linear Model的不同主要是学习过程的不同
      <ul>
        <li>目标函数改变导致梯度不同，权重更新的公式不同</li>
        <li>权重更新时的细节更多：正则化，模拟退火改变步长</li>
      </ul>
    </ol>
  <li>评价指标</li>准确率 = 正确的标注数 / 总的标注数
  <li>程序</li>
  <ol>
    <li>数据</li>
    训练集：train.conll<br>
    测试集：dev.conll
    <li>代码</li>
    log_linear_model.py:实现Linear Model模型进行词性标注
  </ol>
</ol>
