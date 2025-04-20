"""
穷人的配置工具。可能是个糟糕的主意。使用示例：
$ python train.py config/override_file.py --batch_size=32
这将首先运行config/override_file.py，然后将batch_size覆盖为32

这个文件中的代码将从例如train.py中这样运行：
>>> exec(open('configurator.py').read())

所以它不是一个Python模块，它只是将这段代码从train.py中分离出来
这个脚本中的代码然后会覆盖globals()

我知道人们可能不会喜欢这种方式，我只是真的不喜欢配置的复杂性
以及必须在每个变量前面加上config.。如果有人
想出一个更好的简单Python解决方案，我洗耳恭听。
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        # 假设它是配置文件的名称
        assert not arg.startswith('--')
        config_file = arg
        print(f"使用{config_file}覆盖配置:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # 假设它是一个--key=value参数
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # 尝试对其求值（例如，如果是布尔值、数字等）
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # 如果出错，就直接使用字符串
                attempt = val
            # 确保类型匹配
            assert type(attempt) == type(globals()[key])
            # 祈祷一切顺利
            print(f"覆盖: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"未知配置键: {key}")
