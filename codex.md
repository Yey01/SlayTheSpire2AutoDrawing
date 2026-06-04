# 项目解读：SlayTheSpire2AutoDrawing

## 1. 项目用途

这是一个面向 Windows 桌面的《杀戮尖塔 2》自动绘图/自动涂抹工具。程序通过 Tkinter 提供图形界面，使用 OpenCV 提取图片线稿或识别游戏地图图标，再用 pyautogui 控制鼠标在游戏窗口中拖拽绘制。

项目主要有两种工作模式：

- 常规素描模式：用户选择一张图片，程序用 Canny 边缘检测提取线条路径，然后按屏幕绘制区域比例换算坐标，用鼠标按路径逐段拖拽。
- 迷雾战场模式：程序截取游戏地图区域，识别类似图标/节点的目标块，对每个目标执行螺旋涂抹，并边滚动地图边继续扫描，直到判断到达底部。

程序入口是 `AutoDrawer.py`，已通过 PyInstaller 打包为 `AutoDrawer.exe`，可直接运行 `dist/AutoDrawer.exe` 或项目根目录下的 `AutoDrawer.exe`。

## 2. 项目结构

```text
SlayTheSpire2AutoDrawing-main/
├── AutoDrawer.py          # 主程序，包含 UI、配置、图片解析、鼠标绘制和迷雾模式全部逻辑
├── AutoDrawer.exe         # 已打包的可执行文件
├── AutoDrawer.spec        # PyInstaller 打包配置
├── config.txt             # 本地运行配置
├── icon.ico               # exe 图标
├── README.md              # 原始说明，当前终端读取时存在编码错显
├── README.txt             # 原始文本说明，当前终端读取时存在编码错显
├── test.jpg               # 测试图片
├── dist/
│   ├── AutoDrawer.exe     # PyInstaller 输出的可执行文件
│   └── config.txt         # 打包输出目录内的配置文件
└── build/AutoDrawer/      # PyInstaller 构建中间产物、toc、警告文件等
```

当前没有发现 `requirements.txt`、`pyproject.toml` 等依赖声明文件。依赖需要从 `AutoDrawer.py` 的 import 推断。

## 3. 技术栈与依赖

核心依赖：

- `tkinter` / `ttk`：桌面 GUI、参数输入、滚动面板、弹窗。
- `pyautogui`：获取屏幕尺寸、截图、鼠标移动、按下/松开、滚轮滚动。
- `keyboard`：注册全局快捷键，默认开始 `F9`、暂停/继续 `F8`、停止 `F10`。
- `opencv-python` (`cv2`)：图片边缘检测、轮廓提取、模板匹配、形态学处理。
- `numpy`：图像数组转换和形态学核。
- `Pillow`：把 OpenCV 处理后的预览图显示到 Tkinter Canvas。
- `ctypes` / `sys`：Windows 管理员权限检测和自提权重启。

建议的开发环境安装命令：

```powershell
pip install pyautogui keyboard opencv-python numpy pillow pyinstaller
```

注意：`keyboard` 注册全局热键、`pyautogui` 控制游戏窗口以及程序自身的提权逻辑都偏 Windows 场景。跨平台运行并不是当前项目目标。

## 4. 程序启动流程

`AutoDrawer.py` 的入口逻辑位于文件末尾：

1. 调用 `ctypes.windll.shell32.IsUserAnAdmin()` 检查管理员权限。
2. 如果已有管理员权限，创建 `tk.Tk()` 根窗口并实例化 `AutoSketchApp`。
3. 如果没有管理员权限，使用 `ShellExecuteW(..., "runas", ...)` 以管理员权限重新启动当前 Python 进程，然后退出原进程。

管理员权限的目的主要是保证鼠标控制和全局快捷键在游戏窗口上可靠生效。

## 5. 主类与状态

项目的核心类是 `AutoSketchApp`。它集中管理以下内容：

- UI 组件和 Tkinter 变量。
- 当前图片路径、图片尺寸、提取出的轮廓路径。
- 运行状态：`is_running`、`is_paused`、`stop_requested`。
- 配置文件路径：固定为 `config.txt`。
- 快捷键状态：默认 `F9` 开始、`F8` 暂停/继续、`F10` 停止。

程序运行时会把耗时绘制任务放到后台线程中，避免 Tkinter 界面完全卡死：

- 常规绘制：`threading.Thread(target=self.draw_task, daemon=True)`
- 迷雾模式：`threading.Thread(target=self.mist_mode_task, daemon=True)`

UI 更新从后台线程返回主线程时，主要通过 `self.root.after(...)` 调度。

## 6. UI 构成

`setup_ui()` 构建了一个可滚动的设置面板，主要分区如下：

- 图片设置：选择图片、显示边缘检测预览、调整线条阈值和短线过滤。
- 绘制区域：用百分比设置屏幕左、右、上、下避让区域，也支持全屏半透明遮罩手动框选区域。
- 绘制参数：鼠标拖拽步长、起落笔延迟、鼠标左/右键、暂停恢复时自动对齐。
- 迷雾战场设置：涂抹外扩大小、螺旋线间距、螺旋绘制步幅、单次下滚距离。
- 快捷键设置：开始、暂停/继续、停止三个热键支持在界面中修改。
- 操作按钮：启动迷雾战场、保存当前设置。

窗口默认置顶，尺寸为 `520x750`，最小尺寸为 `450x400`。

## 7. 配置文件

配置文件是 `config.txt`，使用 `configparser` 读取，保存时直接覆盖整个文件。主要分区如下：

```ini
[线条设置]
threshold = 100
min_len = 10

[绘制区域]
left_margin = 16
right_margin = 19
top_margin = 10
bottom_margin = 7

[绘制参数]
drag_step = 5
delay = 0.01
mouse_btn = right
auto_align = True

[迷雾设置]
mist_margin = 25
mist_spacing = 1
mist_step = 15
mist_scroll = 1500
```

参数含义：

- `threshold`：Canny 边缘检测低阈值，高阈值为它的 2 倍。值越低，提取出的线条越多。
- `min_len`：过滤过短轮廓，减少噪点。
- `left_margin/right_margin/top_margin/bottom_margin`：屏幕绘制区域的边距百分比。
- `drag_step`：鼠标沿路径插值移动的步长。越大速度越快，但线条可能更粗糙或断裂。
- `delay`：起笔、落笔、恢复绘制时的延迟。过低可能导致游戏没有及时响应鼠标状态变化。
- `mouse_btn`：用于绘制的鼠标按键，支持 `left` 或 `right`。
- `auto_align`：暂停恢复时是否使用截图模板匹配寻找原地图位置并拖动对齐。
- `mist_margin`：迷雾模式中识别到图标后的涂抹外扩半径。
- `mist_spacing`：螺旋涂抹线距，越小越密。
- `mist_step`：螺旋前进步幅，越大越快但可能漏涂。
- `mist_scroll`：迷雾模式每轮向下滚动的距离。

代码中的默认配置和当前 `config.txt` 有细微差异，例如代码默认 `top_margin = 9`、`delay = 0.02`、迷雾参数为 `40/3/20/1200`，而现有配置文件中为 `10/0.01/25/1/15/1500`。实际运行优先读取现有 `config.txt`。

## 8. 常规素描模式流程

常规模式由 `load_image()`、`update_preview()`、`start_drawing()` 和 `draw_task()` 串起来：

1. 用户选择图片。
2. `cv2.imdecode(np.fromfile(...), cv2.IMREAD_COLOR)` 读取图片，兼容中文路径。
3. 转灰度图，执行 `cv2.Canny(gray, threshold, threshold * 2)` 提取边缘。
4. 使用 `cv2.findContours(..., cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)` 提取轮廓。
5. 用 `cv2.arcLength` 过滤短线，再用 `cv2.approxPolyDP` 简化路径。
6. 对轮廓按“离当前点最近”的贪心策略排序，并在必要时反转路径，减少鼠标空跑距离。
7. 预览区域显示边缘图，并提示提取到的路径段数量。
8. 按 `F9` 后进入 `draw_task()`，计算屏幕绘制区域、图片缩放比例和居中偏移。
9. 对每条轮廓逐点换算为屏幕坐标，使用 `pyautogui.mouseDown/moveTo/mouseUp` 完成拖拽绘制。
10. 长线段会按 `drag_step` 插值拆成多个中间点，避免鼠标移动跨度过大。

常规模式不会直接理解游戏内部数据，它只是把图片线稿转换为屏幕鼠标轨迹。因此绘制效果依赖游戏窗口位置、地图缩放、鼠标按钮、帧率和配置参数。

## 9. 暂停、继续与自动对齐

暂停由快捷键控制，内部只是切换 `is_paused` 状态。绘制循环会在每个关键点调用检查函数：

- 常规模式：`check_pause(target_x, target_y, should_be_down)`
- 迷雾模式：`check_mist_pause(resume_x=None, resume_y=None, resume_down=False)`

如果启用了 `auto_align`：

1. 暂停时截取绘制区域中央约 50% 大小作为锚点图。
2. 恢复时在绘制区域内截图，用 `cv2.matchTemplate(..., cv2.TM_CCOEFF_NORMED)` 查找锚点。
3. 匹配度超过阈值后，通过鼠标左键物理拖动地图进行校准。
4. 常规模式会维护 `global_offset_x/global_offset_y`，把对齐造成的位移补偿到后续绘制坐标上。

这个机制用于处理暂停期间用户或游戏滚动了地图，导致原绘制坐标偏移的问题。

## 10. 迷雾战场模式流程

迷雾模式由 `on_btn_mist()` 启动，核心逻辑在 `mist_mode_task()`。

整体过程：

1. 根据绘制区域边距计算屏幕扫描框。
2. 右键点击扫描框中心获取游戏焦点。
3. 连续向上滚动，尽量把地图移动到顶部。
4. 对当前屏幕截图执行图标识别。
5. 对识别出的目标中心进行螺旋涂抹。
6. 截取底部一段作为滚动锚点，向下滚动地图。
7. 用模板匹配计算实际滚动距离，用于全局去重。
8. 重复扫描和涂抹，直到判断触底或锚点匹配失败。

图标识别逻辑：

- 截图转灰度。
- 使用大核高斯模糊得到背景趋势。
- 用 `cv2.subtract(blur, gray)` 得到局部高频差异。
- 二值化后做开运算、闭运算和膨胀，把图标聚成块。
- `cv2.findContours(..., cv2.RETR_EXTERNAL, ...)` 查找外轮廓。
- 用宽高、长宽比、填充率过滤候选目标。
- 通过 `painted_nodes` 和全局 Y 坐标去重，避免滚动后重复涂同一节点。

涂抹逻辑：

- 目标中心为螺旋中心。
- 半径由 `max(w, h) * 0.55 + mist_margin` 计算。
- `mist_spacing` 控制螺旋线距。
- `mist_step` 控制角度增量，也就是涂抹速度和稠密度。

为减少漏涂，代码会跳过尚未触底时贴近扫描框底部的半截图标，等待下一次滚动后图标完整进入屏幕再处理。

## 11. 快捷键机制

快捷键由 `keyboard.add_hotkey()` 注册，当前设计支持：

- 开始常规素描：默认 `F9`
- 暂停/继续：默认 `F8`
- 停止：默认 `F10`

界面中可以点击“修改”后捕获按键。捕获时会临时清除已注册热键，避免冲突。程序会校验三个快捷键不能重复；注册失败时会回滚到旧配置。

关闭窗口时调用 `keyboard.unhook_all()` 清理全局热键。

## 12. 打包方式

项目包含 `AutoDrawer.spec`，这是 PyInstaller 的打包配置：

- 入口脚本：`AutoDrawer.py`
- 输出程序名：`AutoDrawer`
- 窗口模式：`console=False`
- 图标：`icon.ico`
- UPX：启用

可用以下命令重新打包：

```powershell
pyinstaller AutoDrawer.spec
```

打包产物位于 `dist/AutoDrawer.exe`。`build/AutoDrawer/warn-AutoDrawer.txt` 中的很多缺失模块是 PyInstaller 对可选平台模块的提示，不一定代表程序运行失败。例如 `posix`、`AppKit`、`Quartz`、`Xlib` 等多属于非 Windows 平台或可选依赖。

## 13. 维护注意事项

- 当前业务逻辑全部集中在 `AutoDrawer.py`，单文件超过 1000 行。后续如果继续扩展，建议拆分为 UI、配置、图片处理、鼠标绘制、迷雾识别几个模块。
- 项目没有依赖清单，建议补充 `requirements.txt`，否则换机器运行时只能从 import 推断依赖。
- README 文件在当前终端读取时显示乱码，建议重新保存为 UTF-8，以便后续维护者正常阅读。
- 自动鼠标控制会真实操作当前屏幕，调试时应避免误点其它窗口。
- `pyautogui.PAUSE = 0` 会尽可能取消全局动作间隔，速度更快，但也更依赖游戏响应和机器性能。
- 常规模式和迷雾模式都依赖屏幕截图和模板匹配，游戏缩放、窗口遮挡、帧率和地图视觉变化都会影响成功率。
- 程序运行时会请求管理员权限，这对用户体验和安全提示都有影响；如果后续做分发，需要明确说明原因。

## 14. 总结

这个项目本质上是一个“屏幕视觉识别 + 鼠标自动化”的 Windows 桌面工具。常规素描模式把外部图片转换成鼠标轨迹，迷雾战场模式则直接从游戏画面中识别节点并执行涂抹。它没有复杂的后端或数据模型，核心风险集中在图像识别阈值、屏幕坐标换算、鼠标时序、管理员权限和游戏窗口状态。
