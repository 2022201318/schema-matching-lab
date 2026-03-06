数据描述：
source1-2：benchmark CheMBL源表和目标表。数据属性：生物化学。
source3-4：benchmark Opendata源表和目标表。数据属性：公共政策/经济发展/财务审计。
source5-6：benchmark TCP-DI源表和目标表。数据属性：商业/消费者。
source7-8：benchmark TCP-DI源表和目标表。数据属性：人文艺术。
这些数据中，s_metadata(数据源级元数据)是人为标注的，t_metadata（表级元数据）源自benchmark本身。
对应ground truth中的answer1-4，并在answer中补全了数据源信息。
source9-18：benchmark GDC源表和目标表。数据属性：医学/临床数据。
这些数据中，s_metadata源于数据源引用的文章数据，t_metadata缺失。
对应ground truth中的answer5，answer中以GDB目标表名为KEY，整合了10张表的匹配关系。
未来可能会切割数据模拟一源多表。

实验阶段一:
对数据源级别、表级别、列级别分别进行语义语义增强，输出为语义视图。
对于数据源级别：
语义视图包括：数据源名称，数据源描述，包含的表名，每个表包含的列。
如果本身有描述，就沿用，使用规整，不调用模型。
如果没有，调用Qwen-7b小模型，输入表、列，生成描述。
对于表级别，调用Qwen-7b，输入列、列样本，列属性
