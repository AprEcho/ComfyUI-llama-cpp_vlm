import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";
import { LlamaStudioUI } from "./ls_ui.js";

// Llama Prompt Studio - A Professional Prompt Engineering Workspace
app.registerExtension({
    name: "Llama.PromptStudio",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LlamaOmniTaskPrompter") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // 1. 添加提示词工作室按钮
                this.addWidget("button", "🎨 Llama Prompt Studio", null, () => {
                    LlamaStudio.open(this);
                });

                // 2. 绑定权重调整功能
                // 延时等待 DOM 渲染完成，确保能抓到 textarea
                setTimeout(() => {
                    const promptWidget = this.widgets.find(w => w.name === "user_prompt (意图/灵感)");
                    if (promptWidget && promptWidget.inputEl) {
                        this.setupWeightAdjustment(promptWidget.inputEl);
                    }
                }, 100);

                return r;
            };

            // 权重调整核心逻辑
            nodeType.prototype.setupWeightAdjustment = function (inputEl) {
                const adjustWeight = (delta) => {
                    const start = inputEl.selectionStart;
                    const end = inputEl.selectionEnd;
                    let text = inputEl.value;

                    // 获取当前光标位置的单词或选中的文本
                    let selectedText = text.substring(start, end);
                    let rangeStart = start;
                    let rangeEnd = end;

                    // 如果没有选中文本，尝试自动捕捉光标下的单词
                    if (start === end) {
                        // 向前找
                        const textBefore = text.substring(0, start);
                        const beforeMatch = textBefore.match(/[a-zA-Z0-9_.\u4e00-\u9fa5\s]+$/);
                        // 向后找
                        const textAfter = text.substring(start);
                        const afterMatch = textAfter.match(/^[a-zA-Z0-9_.\u4e00-\u9fa5\s]+/);
                        
                        if (beforeMatch || afterMatch) {
                            rangeStart = start - (beforeMatch ? beforeMatch[0].length : 0);
                            rangeEnd = start + (afterMatch ? afterMatch[0].length : 0);
                            selectedText = text.substring(rangeStart, rangeEnd).trim();
                        }
                    }

                    if (!selectedText) return;

                    // 正则：匹配 (text:weight)
                    const weightRegex = /^\((.*):([0-9.]+)\)$/;
                    const match = selectedText.match(weightRegex);

                    let replacement;
                    if (match) {
                        const content = match[1];
                        let weight = parseFloat(match[2]);
                        weight = Math.max(0, parseFloat((weight + delta).toFixed(2)));
                        replacement = `(${content}:${weight})`;
                    } else {
                        // 如果是纯文本，包裹并设置初始权重
                        let weight = Math.max(0, parseFloat((1.0 + delta).toFixed(2)));
                        replacement = `(${selectedText}:${weight})`;
                    }

                    const newText = text.substring(0, rangeStart) + replacement + text.substring(rangeEnd);
                    inputEl.value = newText;
                    
                    // 保持选中状态
                    inputEl.setSelectionRange(rangeStart, rangeStart + replacement.length);
                    
                    // 触发 ComfyUI 内部更新
                    if (inputEl.oninput) inputEl.oninput();
                    const widget = this.widgets.find(w => w.inputEl === inputEl);
                    if (widget) widget.value = newText;
                    this.setDirtyCanvas(true, true);
                };

                // A. 鼠标 Alt + 滚轮 (避开浏览器 Ctrl+滚轮 缩放冲突)
                inputEl.addEventListener("wheel", (e) => {
                    if (!e.altKey) return;
                    e.preventDefault();
                    const delta = e.deltaY > 0 ? -0.05 : 0.05;
                    adjustWeight(delta);
                }, { passive: false });

                // B. 键盘 Ctrl + 方向键 (行业通用快捷键)
                inputEl.addEventListener("keydown", (e) => {
                    if (e.ctrlKey && (e.key === "ArrowUp" || e.key === "ArrowDown")) {
                        e.preventDefault();
                        const delta = e.key === "ArrowUp" ? 0.05 : -0.05;
                        adjustWeight(delta);
                    }
                });
            };
        }
    }
});

const LlamaStudio = {
    window: null,
    activeNode: null,
    currentTab: "browser",
    currentCategory: "",
    promptData: {},
    stagePos: [], 
    stageNeg: [], 
    lastClickedIndex: -1,
    displayedTags: [], // Track flat list of tags in current grid for Shift+Click
    isEditMode: false,
    isEditingPreset: false,
    history: [],
    config: {
        width: 1000,
        height: 800,
        left: 100,
        top: 50,
        opacity: 0.95
    },

    async open(node) {
        this.activeNode = node;
        await this.loadData();
        this.syncFromNode();
        if (!this.window) {
            this.createWindow();
        }
        this.window.style.display = "flex";
        this.refresh();
    },

    async loadData() {
        try {
            const response = await fetch("/llama-cpp-vlm/prompts");
            const data = await response.json();
            if (data.error) {
                console.error("LlamaStudio: Backend error", data.error);
                return;
            }
            this.promptData = data;
            if (!this.currentCategory && Object.keys(this.promptData).length > 0) {
                this.currentCategory = Object.keys(this.promptData)[0];
            }
        } catch (e) {
            console.error("LlamaStudio: Load data failed", e);
        }

        const savedHistory = localStorage.getItem("llama_prompt_studio_history");
        if (savedHistory) {
            try { this.history = JSON.parse(savedHistory); } catch(e) {}
        }
    },
    syncFromNode() {
        const getVal = (name) => {
            const w = this.activeNode.widgets.find(w => w.name && w.name.startsWith(name));
            return w && w.value ? w.value.split(", ").filter(t => t.trim()) : [];
        };
        this.stagePos = getVal("library_tags");
        this.stageNeg = getVal("negative_library_tags");
    },

    applyToNode() {
        if (!this.activeNode) return;
        const setVal = (name, tags) => {
            const w = this.activeNode.widgets.find(w => w.name && w.name.startsWith(name));
            if (w) w.value = tags.join(", ");
        };
        setVal("library_tags", this.stagePos);
        setVal("negative_library_tags", this.stageNeg);
        this.activeNode.setDirtyCanvas(true, true);
        
        if (this.stagePos.length > 0 || this.stageNeg.length > 0) {
            this.history.unshift({ time: new Date().toLocaleString(), pos: [...this.stagePos], neg: [...this.stageNeg] });
            if (this.history.length > 20) this.history.pop();
            localStorage.setItem("llama_prompt_studio_history", JSON.stringify(this.history));
        }
    },

    createWindow() {
        LlamaStudioUI.injectStyle();

        this.window = $el("div.llama-studio-window", {

            style: {
                width: this.config.width + "px", height: this.config.height + "px",
                left: this.config.left + "px", top: this.config.top + "px",
                opacity: this.config.opacity, resize: "both"
            }
        }, [
            // Header
            $el("div.llama-studio-header", [
                $el("div", { style: { display: "flex", alignItems: "center", gap: "20px" } }, [
                    $el("span", { textContent: "Llama Prompt Studio", style: { fontWeight: "bold", color: "#00bcd4", fontSize: "16px" } }),
                    $el("div.ls-tab-bar", [
                        $el("span.ls-tab-btn.active", { textContent: "词库浏览器", onclick: (e) => this.switchTab("browser", e.target) }),
                        $el("span.ls-tab-btn", { textContent: "结构化构思", onclick: (e) => this.switchTab("slots", e.target) }),
                        $el("span.ls-tab-btn", { textContent: "预设管理", onclick: (e) => this.switchTab("presets", e.target) }),
                        $el("span.ls-tab-btn", { textContent: "历史记录", onclick: (e) => this.switchTab("history", e.target) })
                    ]),
                    $el("input", {
                        placeholder: "🔍 全库模糊搜索...",
                        style: { padding: "6px 15px", borderRadius: "15px", border: "1px solid #555", backgroundColor: "#111", color: "#eee", width: "180px" },
                        oninput: (e) => this.handleSearch(e.target.value)
                    })
                ]),
                $el("div", { style: { display: "flex", alignItems: "center", gap: "10px" } }, [
                    $el("span", { textContent: "透明度:", style: { fontSize: "11px", color: "#888" } }),
                    $el("input", {
                        type: "range", min: "0.2", max: "1.0", step: "0.05", value: this.config.opacity,
                        style: { width: "60px" },
                        oninput: (e) => { this.window.style.opacity = e.target.value; this.config.opacity = e.target.value; }
                    }),
                    $el("button", { textContent: "✕", style: { background: "none", color: "#666", border: "none", fontSize: "20px", cursor: "pointer" }, onclick: () => this.window.style.display = "none" })
                ])
            ]),

            // Body
            $el("div.llama-studio-body", [
                $el("div.llama-studio-sidebar"),
                $el("div.llama-studio-content", [
                    $el("div.llama-studio-grid")
                ])
            ]),

            // Dock (Staging Area)
            $el("div.llama-studio-dock", [
                $el("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center" } }, [
                    $el("div", { style: { display: "flex", gap: "10px", alignItems: "center", flex: 1, overflow: "hidden" } }, [
                        $el("span", { textContent: "正向提示词", style: { color: "#4caf50", fontSize: "12px", fontWeight: "bold", flexShrink: 0 } }),
                        $el("div.ls-stage-container-pos", { style: { display: "flex", flexWrap: "wrap", gap: "5px", overflowY: "auto", maxHeight: "60px" } })
                    ]),
                    $el("button", { 
                        textContent: "确认", 
                        style: { padding: "8px 25px", backgroundColor: "#00838f", color: "white", border: "none", borderRadius: "6px", cursor: "pointer", fontWeight: "bold", marginLeft: "15px", flexShrink: 0, boxShadow: "0 4px 10px rgba(0,0,0,0.3)" },
                        onclick: () =>{ 
                            this.applyToNode()
                            this.window.style.display = "none"; // 2. 执行完后关闭窗口
                        }
                    })
                ]),
                $el("div", { style: { display: "flex", gap: "10px", alignItems: "center" } }, [
                    $el("span", { textContent: "负面提示词", style: { color: "#e91e63", fontSize: "12px", fontWeight: "bold", flexShrink: 0 } }),
                    $el("div.ls-stage-container-neg", { style: { display: "flex", flexWrap: "wrap", gap: "5px", overflowY: "auto", maxHeight: "40px" } })
                ])
            ])
        ]);

        document.body.appendChild(this.window);
        this.setupInteractivity();
    },

    setupInteractivity() {
        const header = this.window.querySelector(".llama-studio-header");
        let isDragging = false, startX, startY, startLeft, startTop;

        header.onmousedown = (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON') return;
            isDragging = true;
            startX = e.clientX; startY = e.clientY;
            startLeft = this.window.offsetLeft; startTop = this.window.offsetTop;
            document.addEventListener("mousemove", onMouseMove);
            document.addEventListener("mouseup", onMouseUp);
        };

        const onMouseMove = (e) => {
            if (!isDragging) return;
            this.config.left = startLeft + (e.clientX - startX);
            this.config.top = startTop + (e.clientY - startY);
            this.window.style.left = this.config.left + "px";
            this.window.style.top = this.config.top + "px";
        };

        const onMouseUp = () => { isDragging = false; document.removeEventListener("mousemove", onMouseMove); };

        const observer = new ResizeObserver(entries => {
            for (let entry of entries) {
                this.config.width = entry.contentRect.width;
                this.config.height = entry.contentRect.height;
            }
        });
        observer.observe(this.window);
    },

    switchTab(tab, el) {
        this.currentTab = tab;
        this.window.querySelectorAll(".ls-tab-btn").forEach(b => b.classList.remove("active"));
        el.classList.add("active");
        this.refresh();
    },

    handleSearch(query) {
        if (!query) { this.currentCategory = Object.keys(this.promptData)[0]; this.refresh(); return; }
        query = query.toLowerCase();
        this.lastSearchResults = { "搜索结果": {} };
        const scan = (obj) => {
            for (const [k, v] of Object.entries(obj)) {
                if (v && typeof v === "object") scan(v);
                else if (k.toLowerCase().includes(query) || (v && v.toString().toLowerCase().includes(query))) this.lastSearchResults["搜索结果"][k] = v;
            }
        };
        scan(this.promptData);
        this.currentCategory = "SEARCH";
        this.currentTab = "browser"; // Switch to browser on search
        this.refresh();
    },

    refresh() {
        this.renderSidebar();
        this.renderContent();
        this.renderDock();
    },

    renderSidebar() {
        const sidebar = this.window.querySelector(".llama-studio-sidebar");
        sidebar.innerHTML = "";
        
        if (this.currentTab === "browser") {
            const cats = this.currentCategory === "SEARCH" ? ["SEARCH"] : Object.keys(this.promptData);
            cats.forEach(cat => {
                const btn = $el("button.ls-btn-cat", {
                    textContent: cat === "SEARCH" ? "🔍 搜索结果" : cat,
                    onclick: () => { this.currentCategory = cat; this.refresh(); }
                });
                if (this.currentCategory === cat) btn.classList.add("active");
                sidebar.appendChild(btn);
            });
        } else if (this.currentTab === "slots") {
            sidebar.appendChild($el("div", { textContent: "💡 结构化构思模式", style: { padding: "10px", fontSize: "12px", color: "#888", textAlign: "center" } }));
            ["主体内容", "构图/镜头", "光影/色彩", "画风/艺术", "负面修正"].forEach(s => {
                sidebar.appendChild($el("div", { textContent: s, style: { padding: "8px 15px", fontSize: "13px", borderLeft: "3px solid #00838f", background: "#1a1a1a", color: "#ccc" } }));
            });
        }
    },

    renderContent() {
        const grid = this.window.querySelector(".llama-studio-grid");
        grid.innerHTML = "";
        this.displayedTags = [];

        if (this.currentTab === "browser") {
            const data = this.currentCategory === "SEARCH" ? this.lastSearchResults["搜索结果"] : this.promptData[this.currentCategory];
            if (!data) return;
            const isNegCat = this.currentCategory.includes("NEGATIVE") || this.currentCategory.includes("负面");
            this.renderItems(data, isNegCat, grid);
        } else if (this.currentTab === "presets") {
            grid.style.display = "flex";
            grid.style.flexDirection = "column";
            grid.style.gap = "10px";
            this.renderPresetsTab(grid);
        } else if (this.currentTab === "slots") {
            grid.style.display = "flex";
            grid.style.flexDirection = "column";
            this.renderSlotsTab(grid);
        }
        
        // Reset grid display if switching back
        if (this.currentTab === "browser") {
            grid.style.display = "grid";
        }
    },

    renderItems(obj, isNegCat, container) {
        // 辅助函数：判断标签是否已被选中（兼容带权重格式）
        const isSelected = (key, stage) => {
            return stage.some(t => t === key || t.startsWith(`(${key}:`));
        };

        for (const [key, value] of Object.entries(obj)) {
            if (typeof value === "object" && value !== null) {
                container.appendChild($el("div", { textContent: key, style: { gridColumn: "1 / -1", color: "#00bcd4", fontSize: "12px", padding: "15px 0 5px 0", borderBottom: "1px solid #333", fontWeight: "bold" } }));
                this.renderItems(value, isNegCat, container);
            } else {
                const index = this.displayedTags.length;
                this.displayedTags.push({ key, value, isNeg: isNegCat });
                
                const card = $el("div.ls-tag-card", {
                    onclick: (e) => this.handleTagClick(index, e),
                    dataset: { index: index }
                }, [
                    $el("span", { textContent: key, style: { fontWeight: "bold", color: isNegCat ? "#ff4081" : "#ffd700", fontSize: "13px" } }),
                    $el("span", { textContent: value, style: { fontSize: "10px", opacity: "0.5", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" } })
                ]);
                
                if (isSelected(key, this.stagePos)) card.classList.add("selected-pos");
                if (isSelected(key, this.stageNeg)) card.classList.add("selected-neg");
                
                container.appendChild(card);
            }
        }
    },

    handleTagClick(index, event) {
        const item = this.displayedTags[index];
        const isNeg = item.isNeg;
        const stage = isNeg ? this.stageNeg : this.stagePos;
        
        // 查找是否存在该标签（包括带权重的版本）
        const existingIndex = stage.findIndex(t => t === item.key || t.startsWith(`(${item.key}:`));
        const isCurrentlySelected = existingIndex !== -1;

        if (event.shiftKey && this.lastClickedIndex !== -1) {
            const start = Math.min(this.lastClickedIndex, index);
            const end = Math.max(this.lastClickedIndex, index);
            
            for (let i = start; i <= end; i++) {
                const batchItem = this.displayedTags[i];
                const bStage = batchItem.isNeg ? this.stageNeg : this.stagePos;
                const bIdx = bStage.findIndex(t => t === batchItem.key || t.startsWith(`(${batchItem.key}:`));
                
                if (isCurrentlySelected) {
                    if (bIdx > -1) bStage.splice(bIdx, 1);
                } else {
                    if (bIdx === -1) bStage.push(batchItem.key);
                }
            }
        } else {
            if (isCurrentlySelected) stage.splice(existingIndex, 1);
            else stage.push(item.key);
        }

        this.lastClickedIndex = index;
        this.refresh();
    },

    renderPresetsTab(container) {
        container.appendChild($el("h3", { textContent: "📋 我的预设组合", style: { margin: "0 0 10px 0", color: "#00bcd4" } }));

        // Find all preset-like categories (containing 'My Presets' or from preset path)
        for (const [catName, tags] of Object.entries(this.promptData)) {
            if (catName.includes("My Presets") || catName.includes("预设") || catName.includes("characters") || catName.includes("scenes")) {
                for (const [name, val] of Object.entries(tags)) {
                    if (typeof val !== "string") continue;

                    const item = $el("div.ls-preset-item", [
                        $el("div", [
                            $el("div", { textContent: name, style: { fontWeight: "bold", fontSize: "14px", color: "#fff" } }),
                            $el("div", { textContent: `分类: ${catName.replace("[预设] ", "")}`, style: { fontSize: "11px", color: "#888" } })
                        ]),
                        $el("div", { style: { display: "flex", gap: "6px" } }, [
                            $el("button", {
                                textContent: "🚀 快速应用", style: { padding: "4px 8px", background: "#00838f", border: "none", color: "#fff", borderRadius: "3px", cursor: "pointer", fontSize: "11px" },
                                onclick: () => {
                                    this.loadPresetToStage(name, catName, val, false);
                                    this.applyToNode();
                                    this.refresh();
                                }
                            }),
                            $el("button", {
                                textContent: "✏️ 载入修改", style: { padding: "4px 8px", background: "#f57c00", border: "none", color: "#fff", borderRadius: "3px", cursor: "pointer", fontSize: "11px" },
                                onclick: () => {
                                    this.loadPresetToStage(name, catName, val, true);
                                    this.currentTab = "browser";
                                    this.refresh();
                                }
                            }),
                            $el("button", {
                                textContent: "🗑", style: { padding: "4px 8px", background: "#c62828", border: "none", color: "#fff", borderRadius: "3px", cursor: "pointer", fontSize: "11px", title: "删除预设" },
                                onclick: async () => {
                                    if(confirm(`确定删除预设 [${name}] 吗？`)) {
                                        try {
                                            const response = await fetch("/llama-cpp-vlm/presets/delete", {
                                                method: "POST",
                                                headers: { "Content-Type": "application/json" },
                                                body: JSON.stringify({ name: name, category: catName.replace("[预设] ", "") })
                                            });
                                            if (response.ok) {
                                                await this.loadData();
                                                this.refresh();
                                            } else {
                                                const err = await response.json();
                                                alert("删除失败: " + (err.error || response.statusText));
                                            }
                                        } catch(e) { alert("删除请求失败: " + e); }
                                    }
                                }
                            })
                        ])
                    ]);
                    container.appendChild(item);
                }
            }
        }
    },

    loadPresetToStage(name, catName, val, setEditMode = false) {
        if (val.includes(" ||| ")) {
            const parts = val.split(" ||| ");
            this.stagePos = parts[0].split(", ").map(t=>t.trim()).filter(t => t);
            this.stageNeg = parts[1].split(", ").map(t=>t.trim()).filter(t => t);
        } else {
            this.stagePos = val.split(", ").map(t=>t.trim()).filter(t => t);
            this.stageNeg = [];
        }
        this.lastPresetName = name;
        this.lastPresetCat = catName.replace("[预设] ", "");
        this.isEditingPreset = setEditMode;
    },
    renderHistoryTab(container) {
        container.appendChild($el("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" } }, [
            $el("h3", { textContent: "🕒 最近同步历史 (Top 20)", style: { margin: "0", color: "#00bcd4" } }),
            $el("button", { 
                textContent: "🗑 清空历史", style: { padding: "4px 10px", background: "#c62828", border: "none", color: "#fff", borderRadius: "4px", cursor: "pointer", fontSize: "11px" },
                onclick: () => { if(confirm("清空所有历史记录?")) { this.history = []; localStorage.removeItem("llama_prompt_studio_history"); this.refresh(); } }
            })
        ]));
        
        if (this.history.length === 0) {
            container.appendChild($el("div", { textContent: "暂无同步记录", style: { color: "#888", textAlign: "center", padding: "20px" } }));
            return;
        }

        this.history.forEach((item, index) => {
            const histEl = $el("div.ls-preset-item", { style: { flexDirection: "column", alignItems: "stretch", gap: "8px" } }, [
                $el("div", { style: { display: "flex", justifyContent: "space-between", borderBottom: "1px solid #444", paddingBottom: "5px" } }, [
                    $el("span", { textContent: item.time, style: { fontSize: "12px", color: "#aaa" } }),
                    $el("div", { style: { display: "flex", gap: "8px" } }, [
                        $el("button", { 
                            textContent: "加载", style: { padding: "3px 10px", background: "#00838f", border: "none", color: "#fff", borderRadius: "3px", cursor: "pointer", fontSize: "11px" },
                            onclick: () => { this.stagePos = [...item.pos]; this.stageNeg = [...item.neg]; this.refresh(); }
                        }),
                        $el("button", { 
                            textContent: "✕", style: { padding: "3px 8px", background: "transparent", border: "1px solid #666", color: "#ccc", borderRadius: "3px", cursor: "pointer", fontSize: "11px" },
                            onclick: () => { this.history.splice(index, 1); localStorage.setItem("llama_prompt_studio_history", JSON.stringify(this.history)); this.refresh(); }
                        })
                    ])
                ]),
                $el("div", { style: { fontSize: "12px" } }, [
                    $el("div", { textContent: "正向: " + (item.pos.join(", ") || "无"), style: { color: "#4caf50", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" } }),
                    $el("div", { textContent: "负向: " + (item.neg.join(", ") || "无"), style: { color: "#e91e63", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", marginTop: "3px" } })
                ])
            ]);
            container.appendChild(histEl);
        });
    },

    renderSlotsTab(container) {
        // Find actual category names from promptData
        const allCats = Object.keys(this.promptData);
        const findCats = (keywords) => {
            return allCats.filter(c => keywords.some(k => c.includes(k)));
        };

        const slots = [
            { title: "📌 第 1 步：画质与构图 (Quality & Comp)", cats: findCats(["01", "画质", "构图", "起手"]) },
            { title: "🧍 第 2 步：人物设定 (Character)", cats: findCats(["02", "03", "人物", "发型"]) },
            { title: "👗 第 3 步：服饰与配件 (Clothing)", cats: findCats(["04", "05", "06", "服饰", "饰品"]) },
            { title: "🎬 第 4 步：动作与表情 (Action)", cats: findCats(["07", "动态", "表情", "动作"]) },
            { title: "🌄 第 5 步：环境与道具 (Environment)", cats: findCats(["08", "场景", "环境", "景观"]) },
            { title: "🎨 第 6 步：风格与特效 (Style)", cats: findCats(["09", "艺术", "风格", "色彩", "颜色"]) },
            { title: "🚫 第 7 步：负面修正 (Negative)", cats: findCats(["NEGATIVE", "负面"]) }
        ];

        container.appendChild($el("div", { 
            textContent: "💡 引导式构思：按照下方 1~7 的步骤顺序，点击分类按钮去挑选词条。这能帮您建立结构清晰、主次分明的提示词。", 
            style: { color: "#ccc", fontSize: "12px", marginBottom: "15px", padding: "12px", background: "#222", borderLeft: "4px solid #00bcd4", borderRadius: "4px" } 
        }));

        slots.forEach(slot => {
            if (slot.cats.length === 0) return; // Skip if no matching categories found
            
            const group = $el("div.ls-slot-group", [
                $el("div.ls-slot-title", { textContent: slot.title }),
                $el("div", { style: { display: "flex", flexWrap: "wrap", gap: "10px" } }, 
                    slot.cats.map(catName => {
                        return $el("button.ls-btn-cat", { 
                            textContent: "👉 进入 " + catName, 
                            style: { fontSize: "12px", padding: "8px 15px", background: "#006064", color: "#fff", fontWeight: "bold", border: "1px solid #00838f" },
                            onclick: () => { 
                                this.currentCategory = catName; 
                                this.currentTab = "browser"; 
                                this.refresh(); 
                            }
                        });
                    })
                )
            ]);
            container.appendChild(group);
        });
    },

    renderDock() {
        const renderStage = (container, tags, isNeg) => {
            const el = this.window.querySelector(container);
            el.innerHTML = "";
            tags.forEach((t, index) => {
                const chip = $el("div.ls-stage-chip" + (isNeg ? ".ls-stage-neg" : ".ls-stage-pos"), {
                    textContent: t,
                    title: "滚轮增减权重"
                }, [
                    $el("span", { 
                        textContent: "✕", style: { cursor: "pointer", fontSize: "14px", marginLeft: "8px", fontWeight: "bold" },
                        onclick: (e) => { e.stopPropagation(); tags.splice(index, 1); this.refresh(); }
                    })
                ]);

                // 核心优化：直接滚动气泡调整权重
                chip.addEventListener("wheel", (e) => {
                    e.preventDefault();
                    e.stopPropagation();

                    const delta = e.deltaY > 0 ? -0.05 : 0.05;
                    
                    // 解析当前权重 (tag:1.1)
                    const weightRegex = /^\((.*):([0-9.]+)\)$/;
                    const match = t.match(weightRegex);
                    
                    let newTag;
                    if (match) {
                        const content = match[1];
                        let weight = parseFloat(match[2]);
                        weight = Math.max(0, parseFloat((weight + delta).toFixed(2)));
                        newTag = `(${content}:${weight})`;
                    } else {
                        // 首次包裹
                        let weight = Math.max(0, parseFloat((1.0 + delta).toFixed(2)));
                        newTag = `(${t}:${weight})`;
                    }

                    tags[index] = newTag;
                    this.refresh();
                }, { passive: false });

                el.appendChild(chip);
            });
        };

        renderStage(".ls-stage-container-pos", this.stagePos, false);
        renderStage(".ls-stage-container-neg", this.stageNeg, true);
        
        // Dynamic Dock Actions
        const dock = this.window.querySelector(".llama-studio-dock");
        let btnContainer = dock.querySelector(".ls-dock-actions");
        if (btnContainer) {
            btnContainer.remove();
        }
        btnContainer = $el("div.ls-dock-actions", { style: { display: "flex", justifyContent: "flex-end", gap: "15px", marginTop: "5px", alignItems: "center" } });
        dock.appendChild(btnContainer);
        
        if (this.isEditingPreset && this.lastPresetName) {
             btnContainer.appendChild($el("span", { textContent: `正在编辑预设: [${this.lastPresetName}]`, style: { color: "#f57c00", fontSize: "12px", marginRight: "auto", fontWeight: "bold", background: "#331a00", padding: "4px 10px", borderRadius: "4px" } }));
             btnContainer.appendChild($el("button", { textContent: "💾 覆盖保存", style: { fontSize: "11px", background: "#f57c00", border: "none", color: "#fff", padding: "5px 15px", cursor: "pointer", borderRadius: "4px", fontWeight: "bold" }, onclick: () => this.overwriteCurrentPreset() }));
             btnContainer.appendChild($el("button", { textContent: "❌ 退出编辑", style: { fontSize: "11px", background: "#444", border: "none", color: "#ccc", padding: "5px 12px", cursor: "pointer", borderRadius: "4px" }, onclick: () => { 
                 this.isEditingPreset = false; 
                 this.currentTab = "presets";
                 const tabBtns = this.window.querySelectorAll(".ls-tab-btn");
                 tabBtns.forEach(b => {
                     if (b.textContent === "预设管理") b.classList.add("active");
                     else b.classList.remove("active");
                 });
                 this.refresh(); 
             } }));
        } else {
             btnContainer.appendChild($el("button", { textContent: "💾 保存为 new 预设", style: { fontSize: "11px", background: "#333", border: "1px solid #444", color: "#ccc", padding: "5px 12px", cursor: "pointer", borderRadius: "4px" }, onclick: () => this.saveCurrentAsPreset() }));
             btnContainer.appendChild($el("button", { textContent: "🗑 清空舞台", style: { fontSize: "11px", background: "#333", border: "1px solid #444", color: "#ccc", padding: "5px 12px", cursor: "pointer", borderRadius: "4px" }, onclick: () => { this.stagePos = []; this.stageNeg = []; this.refresh(); } }));
        }
    },

    async overwriteCurrentPreset() {
        if (!this.lastPresetName || !this.lastPresetCat) return;
        try {
            const response = await fetch("/llama-cpp-vlm/presets/save", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    name: this.lastPresetName,
                    category: this.lastPresetCat,
                    positive: this.stagePos.join(", "),
                    negative: this.stageNeg.join(", ")
                })
            });
            if (response.ok) {
                alert("预设已覆盖更新！");
                this.isEditingPreset = false;
                this.currentTab = "presets";
                const tabBtns = this.window.querySelectorAll(".ls-tab-btn");
                tabBtns.forEach(b => {
                    if (b.textContent === "预设管理") b.classList.add("active");
                    else b.classList.remove("active");
                });
                await this.loadData();
                this.refresh();
            } else {
                const err = await response.json();
                alert("保存失败: " + (err.error || response.statusText));
            }
        } catch (e) { alert("覆盖失败: " + e); }
    },

    async saveCurrentAsPreset() {
        const name = prompt("预设名称:", this.lastPresetName || "");
        if (!name) return;
        const cat = prompt("所属分类:", this.lastPresetCat || "我的常用");
        if (!cat) return;

        try {
            const response = await fetch("/llama-cpp-vlm/presets/save", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    name: name,
                    category: cat,
                    positive: this.stagePos.join(", "),
                    negative: this.stageNeg.join(", ")
                })
            });
            if (response.ok) {
                alert("预设保存成功！");
                this.isEditingPreset = false;
                this.currentTab = "presets";
                const tabBtns = this.window.querySelectorAll(".ls-tab-btn");
                tabBtns.forEach(b => {
                    if (b.textContent === "预设管理") b.classList.add("active");
                    else b.classList.remove("active");
                });
                await this.loadData();
                this.refresh();
            } else {
                const err = await response.json();
                alert("保存失败: " + (err.error || response.statusText));
            }
        } catch (e) {
            alert("保存失败: " + e);
        }
    }
};

LlamaStudio.refresh = LlamaStudio.refresh.bind(LlamaStudio);
