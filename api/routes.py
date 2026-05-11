from server import PromptServer
from aiohttp import web
from ..core.common import prompt_lib

@PromptServer.instance.routes.get("/llama-cpp-vlm/prompts")
async def get_prompts_api(request):
    try:
        categories = prompt_lib.get_categories()
        data = {}
        for cat in categories:
            data[cat] = prompt_lib.get_library_data(cat)

        preset_cats = prompt_lib.get_preset_categories()
        for cat in preset_cats:
            preset_name = f"[预设] {cat}" if not cat.startswith("[") else cat
            data[preset_name] = prompt_lib.get_presets(cat)

        return web.json_response(data)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.post("/llama-cpp-vlm/prompts/save")
async def save_prompts_api(request):
    try:
        body = await request.json()
        category = body.get("category")
        data = body.get("data")

        if not category or data is None:
            return web.json_response({"error": "Missing category or data"}, status=400)

        if "[NEGATIVE]" in category:
            return web.json_response({"error": "Cannot edit negatives preset yet."}, status=400)

        success = prompt_lib.save_library_data(category, data)

        if success:
            return web.json_response({"status": "success"})
        else:
            return web.json_response({"error": "Failed to save"}, status=500)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.post("/llama-cpp-vlm/presets/save")
async def save_presets_api(request):
    try:
        body = await request.json()
        name = body.get("name")
        category = body.get("category", "My Presets")
        pos = body.get("positive", "")
        neg = body.get("negative", "")

        if not name:
            return web.json_response({"error": "Missing preset name"}, status=400)

        # Format: "pos ||| neg"
        content = f"{pos} ||| {neg}"
        success = prompt_lib.save_preset(category, name, content)

        if success:
            return web.json_response({"status": "success"})
        else:
            return web.json_response({"error": "Failed to save preset"}, status=500)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.post("/llama-cpp-vlm/presets/delete")
async def delete_presets_api(request):
    try:
        body = await request.json()
        name = body.get("name")
        category = body.get("category")

        if not name or not category:
            return web.json_response({"error": "Missing preset name or category"}, status=400)

        success = prompt_lib.delete_preset(category, name)

        if success:
            return web.json_response({"status": "success"})
        else:
            return web.json_response({"error": "Failed to delete preset"}, status=500)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
