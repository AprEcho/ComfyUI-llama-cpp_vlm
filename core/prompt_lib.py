import os
import json
import random
import glob

class PromptLibrary:
    def __init__(self, base_path):
        self.library_path = os.path.join(base_path, "prompts", "library")
        self.presets_path = os.path.join(base_path, "prompts", "presets")
        self.cache = {}

    def _load_json(self, file_path):
        try:
            # sixgod_prompt uses utf-8-sig (with BOM)
            with open(file_path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {}

    def get_categories(self):
        files = glob.glob(os.path.join(self.library_path, "*.json"))
        return [os.path.basename(f).replace(".json", "") for f in files]
        
    def get_preset_categories(self):
        files = glob.glob(os.path.join(self.presets_path, "*.json"))
        return [os.path.basename(f).replace(".json", "") for f in files]

    def get_library_data(self, category):
        file_path = os.path.join(self.library_path, f"{category}.json")
        if os.path.exists(file_path):
            return self._load_json(file_path)
        return {}

    def save_library_data(self, category, data):
        file_path = os.path.join(self.library_path, f"{category}.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving library data for {category}: {e}")
            return False

    def get_presets(self, type):
        file_path = os.path.join(self.presets_path, f"{type}.json")
        if os.path.exists(file_path):
            return self._load_json(file_path)
        return {}

    def save_preset(self, type, name, content):
        file_path = os.path.join(self.presets_path, f"{type}.json")
        data = self.get_presets(type)
        data[name] = content
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving preset {name}: {e}")
            return False

    def delete_preset(self, type, name):
        file_path = os.path.join(self.presets_path, f"{type}.json")
        data = self.get_presets(type)
        if name in data:
            del data[name]
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                return True
            except Exception as e:
                print(f"Error deleting preset {name}: {e}")
                return False
        return True

    def random_sample(self, categories, count_per_cat=3):
        results = []
        for cat in categories:
            data = self.get_library_data(cat)
            if not data:
                continue
            
            # Flatten the dictionary if it has subcategories
            all_tags = []
            def flatten(d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        flatten(v)
                    else:
                        # If value is string, it's usually English. Key is usually Chinese.
                        # We return the English part for the prompt.
                        all_tags.append((k, v))
            
            flatten(data)
            
            if all_tags:
                sample_count = min(len(all_tags), count_per_cat)
                sampled = random.sample(all_tags, sample_count)
                # Format: EnglishTag (ChineseName) or just EnglishTag
                results.extend([f"{tag[1]}" for tag in sampled])
        
        return ", ".join(results)

    def get_bilingual_sample(self, categories, count_per_cat=3):
        results = []
        for cat in categories:
            data = self.get_library_data(cat)
            if not data:
                continue
            all_tags = []
            def flatten(d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        flatten(v)
                    else:
                        all_tags.append((k, v))
            flatten(data)
            if all_tags:
                sample_count = min(len(all_tags), count_per_cat)
                sampled = random.sample(all_tags, sample_count)
                # Useful for LLM to know what the tags mean in Chinese
                results.extend([f"{tag[1]} ({tag[0]})" for tag in sampled])
        return ", ".join(results)

    def get_full_mapping(self):
        mapping = {}
        categories = self.get_categories()
        for cat in categories:
            data = self.get_library_data(cat)
            self._recursive_build_map(data, mapping)
        
        # Add negatives
        neg_data = self.get_presets("negatives")
        if neg_data:
            self._recursive_build_map(neg_data, mapping)
        return mapping

    def _recursive_build_map(self, data, mapping):
        if not isinstance(data, dict):
            return
        for k, v in data.items():
            if isinstance(v, dict):
                self._recursive_build_map(v, mapping)
            else:
                mapping[k] = v
