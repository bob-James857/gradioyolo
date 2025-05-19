import gradio as gr
from ultralytics import YOLO
import PIL.Image
import os
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 配置 ---
MODEL_PATH = 'yolo11n.pt' # 请确保这是您模型的准确路径
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
# --- /配置 ---

# --- 模型加载 ---
model = None
model_load_error = None

logger.info(f"正在检查模型文件路径: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    model_load_error = f"错误: 模型文件 {MODEL_PATH} 未找到。请检查路径是否正确。"
    logger.error(model_load_error)
else:
    logger.info(f"尝试从 {MODEL_PATH} 加载 YOLO 模型...")
    try:
        model = YOLO(MODEL_PATH)
        logger.info(f"模型 {MODEL_PATH} 加载成功。")
        # 可以尝试用一个虚拟输入来预热模型（可选）
        # try:
        #     dummy_image = PIL.Image.new('RGB', (640, 480), color = 'red')
        #     model.predict(source=dummy_image, verbose=False)
        #     logger.info("模型预热成功。")
        # except Exception as e:
        #     logger.warning(f"模型预热时发生可选错误: {e}")
    except Exception as e:
        model_load_error = f"加载模型时出错: {e}. 请确保 Ultralytics 已正确安装，模型文件有效且路径正确。"
        logger.error(model_load_error)
        model = None # 确保模型变量在出错时为 None
# --- /模型加载 ---

def detect_objects_on_plate(image_pil, confidence_threshold):
    """
    使用 YOLO 模型检测图像中的物体（期望是餐盘上的物体，或餐盘本身，取决于模型训练内容）。

    参数:
    image_pil (PIL.Image.Image): 输入的 PIL 图像。
    confidence_threshold (float): 用于检测的置信度阈值。

    返回:
    PIL.Image.Image: 带有检测框和标签的图像。
    """
    if model is None:
        if model_load_error:
            raise gr.Error(f"模型未能加载: {model_load_error}")
        else:
            raise gr.Error("模型未初始化。请检查服务器日志。")

    if image_pil is None:
        raise gr.Error("未提供输入图像。请上传一张图片。")

    logger.info(f"开始使用置信度 {confidence_threshold} 进行检测...")
    try:
        # Ultralytics YOLO可以直接接受 PIL 图像
        # verbose=False 可以减少控制台输出
        results = model.predict(source=image_pil, conf=confidence_threshold, save=False, verbose=False)
    except Exception as e:
        logger.error(f"模型推理时发生错误: {e}")
        raise gr.Error(f"图像处理时发生错误: {e}")

    if results and len(results) > 0:
        # results[0].plot() 返回一个 NumPy array (BGR格式)
        # 我们需要将其转换回 PIL Image 以便 Gradio 显示
        annotated_image_np = results[0].plot() # 这是 BGR numpy array

        # 将 BGR 转换为 RGB
        # annotated_image_pil = PIL.Image.fromarray(annotated_image_np[:, :, ::-1]) # BGR to RGB
        # .plot() 方法现在默认返回 RGB 格式的 numpy 数组，可以直接转换为 PIL Image
        annotated_image_pil = PIL.Image.fromarray(annotated_image_np)

        logger.info("检测完成，已生成标注图像。")
        return annotated_image_pil
    else:
        logger.info("未检测到任何物体，或结果为空。返回原始图像。")
        # 如果没有检测到任何物体，或者结果为空，返回原始图像
        return image_pil

# --- Gradio 界面 ---
logger.info("正在创建 Gradio 界面...")

# 使用 Blocks API 以获得更灵活的布局和错误处理显示
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
        # YOLO 餐盘检测 Demo
        上传一张图片，模型将会检测并标注出餐盘（或相关物体，取决于您的 `{os.path.basename(MODEL_PATH)}` 模型训练内容）。
        **模型路径:** `{MODEL_PATH}`
        """
    )

    if model_load_error:
        gr.Markdown(f"<h3 style='color:red;'>模型加载失败</h3>\n<p>{model_load_error}</p>")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="上传图片 (Upload Image)", sources=["upload", "webcam", "clipboard"])
            confidence_slider = gr.Slider(minimum=0.05, maximum=1.0, value=DEFAULT_CONFIDENCE_THRESHOLD, step=0.05, label="置信度阈值 (Confidence Threshold)")
            submit_button = gr.Button("开始检测 (Detect)", variant="primary", interactive=model is not None) # 仅当模型加载成功时可交互
        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="检测结果 (Detection Result)")

    submit_button.click(
        fn=detect_objects_on_plate,
        inputs=[input_image, confidence_slider],
        outputs=output_image
    )

    gr.Examples(
        examples=[
            # 在这里可以添加一些示例图片的路径，如果您的环境中有这些图片
            # 例如：["path/to/example_plate1.jpg", 0.5],
            # ["path/to/example_plate2.png", 0.25],
        ],
        inputs=[input_image, confidence_slider], # 确保这里的输入组件与 fn 的输入匹配
        outputs=output_image,
        fn=detect_objects_on_plate, # 指定处理示例的函数
        cache_examples=False # 如果示例图片较大或处理耗时，可以设为True，但需要确保Gradio版本支持
    )

    if model is not None:
        gr.Markdown("模型加载成功，可以开始检测了！")
    else:
        gr.Markdown(f"<p style='color:orange;'>注意：由于模型未能加载，检测功能将不可用。</p>")


if __name__ == '__main__':
    logger.info("正在启动 Gradio 应用...")
    # server_name="0.0.0.0" 使其可以在局域网内通过 IP 地址访问
    share=True 
    demo.launch(server_name="0.0.0.0") # share=True
    # demo.launch() # 仅本地访问