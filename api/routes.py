from server import PromptServer
from aiohttp import web
from ..core.common import prompt_lib

# ==============================================================================
# 提示词词库与预设的 HTTP API 接口
# 这些接口由前端 Prompt Studio (JS) 调用，实现数据的异步读写
# ==============================================================================

@PromptServer.instance.routes.get("/llama-cpp-vlm/prompts")
async def get_prompts_api(request):
    """获取所有提示词分类及其下的标签数据，同时合并预设分类"""
    try:
        categories = prompt_lib.get_categories()
        data = {}
        # 1. 加载常规分类标签
        for cat in categories:
            data[cat] = prompt_lib.get_library_data(cat)

        # 2. 加载预设 (Presets) 分类
        preset_cats = prompt_lib.get_preset_categories()
        for cat in preset_cats:
            preset_name = f"[预设] {cat}" if not cat.startswith("[") else cat
            data[preset_name] = prompt_lib.get_presets(cat)

        return web.json_response(data)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.post("/llama-cpp-vlm/prompts/save")
async def save_prompts_api(request):
    """保存/更新特定分类下的标签列表"""
    try:
        body = await request.json()
        category = body.get("category")
        data = body.get("data")

        if not category or data is None:
            return web.json_response({"error": "缺少分类名称 or 数据内容"}, status=400)

        # 限制：目前不支持通过 API 直接编辑特定的负面预设
        if "[NEGATIVE]" in category:
            return web.json_response({"error": "暂不支持在线编辑负面词预设"}, status=400)

        success = prompt_lib.save_library_data(category, data)

        if success:
            return web.json_response({"status": "success"})
        else:
            return web.json_response({"error": "保存失败"}, status=500)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.post("/llama-cpp-vlm/presets/save")
async def save_presets_api(request):
    """保存新的用户预设 (包含正向和负面词)"""
    try:
        body = await request.json()
        name = body.get("name")
        category = body.get("category", "My Presets")
        pos = body.get("positive", "")
        neg = body.get("negative", "")

        if not name:
            return web.json_response({"error": "预设名称不能为空"}, status=400)

        # 预设文件中的存储格式： "正向词串 ||| 负面词串"
        content = f"{pos} ||| {neg}"
        success = prompt_lib.save_preset(category, name, content)

        if success:
            return web.json_response({"status": "success"})
        else:
            return web.json_response({"error": "保存预设失败"}, status=500)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.post("/llama-cpp-vlm/presets/delete")
async def delete_presets_api(request):
    """删除指定的预设项目"""
    try:
        body = await request.json()
        name = body.get("name")
        category = body.get("category")

        if not name or not category:
            return web.json_response({"error": "缺少预设名称或分类"}, status=400)

        success = prompt_lib.delete_preset(category, name)

        if success:
            return web.json_response({"status": "success"})
        else:
            return web.json_response({"error": "删除预设失败"}, status=500)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
