export const LlamaStudioUI = {
    css: `
        .llama-studio-window {
            position: fixed; z-index: 10001;
            background-color: #1a1a1a; color: #eee;
            border: 1px solid #444; border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.8);
            display: flex; flex-direction: column;
            overflow: hidden; min-width: 600px; min-height: 400px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .llama-studio-header {
            background: linear-gradient(90deg, #222, #333);
            padding: 12px 20px; display: flex; justify-content: space-between; align-items: center;
            cursor: move; border-bottom: 1px solid #444; flex-shrink: 0;
        }
        .llama-studio-body { display: flex; flex: 1; overflow: hidden; min-height: 0; }
        .llama-studio-sidebar {
            width: 220px; background-color: #222; border-right: 1px solid #333;
            display: flex; flex-direction: column; padding: 10px; gap: 5px; overflow-y: auto; flex-shrink: 0;
        }
        .llama-studio-content { flex: 1; display: flex; flex-direction: column; background-color: #1e1e1e; min-width: 0; }
        .llama-studio-grid {
            flex: 1; overflow-y: auto; display: grid;
            grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
            gap: 8px; padding: 15px; align-content: start;
        }
        .llama-studio-dock {
            background-color: #111; border-top: 1px solid #444; padding: 15px;
            display: flex; flex-direction: column; gap: 10px; flex-shrink: 0;
        }
        .ls-btn-cat {
            text-align: left; padding: 8px 12px; background: #333; border: none;
            color: #ccc; border-radius: 6px; cursor: pointer; transition: 0.2s;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        .ls-btn-cat:hover { background: #444; }
        .ls-btn-cat.active { background: #00838f; color: white; box-shadow: 0 0 10px rgba(0,188,212,0.3); }

        .ls-tag-card {
            background: #2a2a2a; border: 1px solid #444; border-radius: 6px;
            padding: 8px; cursor: pointer; display: flex; flex-direction: column;
            gap: 4px; transition: 0.15s; user-select: none;
        }
        .ls-tag-card:hover { transform: translateY(-2px); border-color: #00bcd4; }
        .ls-tag-card.selected-pos { border-color: #4caf50; background: #1b3a1b; }
        .ls-tag-card.selected-neg { border-color: #e91e63; background: #3a1b25; }

        .ls-stage-chip {
            padding: 4px 10px; border-radius: 15px; font-size: 11px;
            display: flex; align-items: center; gap: 6px; cursor: default;
            transition: 0.2s;
        }
        .ls-stage-chip:hover { filter: brightness(1.2); }
        .ls-stage-pos { background-color: #2e7d32; border: 1px solid #4caf50; }
        .ls-stage-neg { background-color: #880e4f; border: 1px solid #e91e63; }

        .ls-tab-bar { display: flex; gap: 10px; margin-right: 20px; }
        .ls-tab-btn { padding: 5px 15px; cursor: pointer; border-bottom: 2px solid transparent; opacity: 0.6; font-weight: bold; }
        .ls-tab-btn.active { opacity: 1; border-bottom-color: #00bcd4; color: #00bcd4; }

        .ls-slot-group {
            margin-bottom: 20px; padding: 10px; background: #252525; border-radius: 8px;
        }
        .ls-slot-title { font-weight: bold; color: #00bcd4; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }

        .ls-preset-item {
            background: #2a2a2a; padding: 12px; border-radius: 8px; border: 1px solid #444;
            display: flex; justify-content: space-between; align-items: center; transition: 0.2s;
        }
        .ls-preset-item:hover { border-color: #00bcd4; background: #333; }

        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #444; border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: #555; }
    `,
    
    injectStyle() {
        if (document.getElementById("llama-studio-style")) return;
        const style = document.createElement("style");
        style.id = "llama-studio-style";
        style.textContent = this.css;
        document.head.appendChild(style);
    }
};
