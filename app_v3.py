import streamlit as st
import sys

# 1. 最先设置页面配置（必须是第一个 Streamlit 命令）
st.set_page_config(page_title="AI 数据分类标注工作流", layout="wide")

# 2. 尝试加载依赖库，如果出错直接显示在界面上
try:
    import pandas as pd
    import json
    import io
    import os
    import re
    import concurrent.futures
    from openai import OpenAI
except Exception as e:
    st.error(f"❌ 依赖库加载失败: {e}")
    st.stop()

# ==========================================
# 辅助函数
# ==========================================

@st.cache_data(show_spinner=False)
def load_excel_sheets(file) -> list[str]:
    """获取 Excel 的所有 sheet 名称"""
    try:
        xl = pd.ExcelFile(file)
        return xl.sheet_names
    except Exception as e:
        st.error(f"无法读取 Sheet: {e}")
        return []

@st.cache_data(show_spinner=False)
def load_excel(file, sheet_name) -> pd.DataFrame | None:
    try:
        return pd.read_excel(file, sheet_name=sheet_name)
    except Exception as e:
        st.error(f"文件读取失败: {e}")
        return None

def call_llm(row, text_col, context_cols, system_prompt, api_key, base_url, model):
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        context_text = "\n".join([f"{col}: {row.get(col, '')}" for col in context_cols])
        user_content = f"""
        【待分析内容】
        {text_col}: {row.get(text_col, '')}
        
        【辅助信息】
        {context_text}
        """
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def parse_json_result(text: str) -> dict:
    try:
        # 1. 尝试直接解析 (去除可能存在的 markdown 代码块标记)
        clean_text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except:
        try:
            # 2. 如果直接解析失败，尝试使用正则提取最外层的 JSON 对象
            # 匹配从第一个 { 到最后一个 } 的内容
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                return json.loads(match.group(0))
        except:
            pass
        
        # 3. 彻底失败，返回原始文本以便调试
        return {"category": "Unknown", "reason": "JSON 解析失败", "raw_output": text}

def stratified_sample(df: pd.DataFrame, label_col: str, frac: float = 0.1) -> pd.DataFrame:
    try:
        return df.groupby(label_col, group_keys=False).apply(
            lambda x: x.sample(frac=frac) if len(x) > 0 else x
        )
    except Exception as e:
        st.error(f"抽样失败 (可能是分类列数据问题): {e}")
        return pd.DataFrame()

def get_ai_reasoning(row, text_col, context_cols, current_category, api_key, base_url, model):
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        context_text = "\n".join([f"{col}: {row.get(col, '')}" for col in context_cols])
        user_content = f"""
        【待分析内容】
        {text_col}: {row.get(text_col, '')}
        
        【辅助信息】
        {context_text}
        
        【当前分类结果】
        {current_category}
        """
        prompt = f"请详细解释为什么这篇内容被归类为“{current_category}”。请结合内容细节给出具体的分析理由。"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个数据分析专家。"},
                {"role": "user", "content": user_content + "\n\n" + prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_misclassification(corrections, processed_data, df, text_col, system_prompt, api_key, base_url, model):
    examples = []
    for idx, correct_label in corrections.items():
        ai_res = processed_data.get(idx, {})
        ai_label = ai_res.get("category", "Unknown")
        content = str(df.loc[idx, text_col])[:200]
        examples.append(f"【案例】\\n内容: {content}...\\nAI原判: {ai_label}\\n人工修正: {correct_label}\\n")
    
    examples_text = "\\n".join(examples)
    prompt = f"""
    你是一个 Prompt 优化专家。以下是 AI 分类错误与人工修正的对比案例：
    {examples_text}
    
    【当前 System Prompt】
    {system_prompt}
    
    请执行以下任务：
    1. 分析误判原因，并输出针对 System Prompt 的具体修改建议（以“建议在 Prompt 中增加规则：......”的格式）。
    2. 根据你的建议，修改并生成一个新的 System Prompt。
    
    请严格按照以下格式输出：
    【分析与建议】
    ...你的分析...
    
    【优化后的 System Prompt】
    ...新的 Prompt 内容...
    """
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def render_prompt_lab(df):
    st.subheader("区域 A: 控制台")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("##### 📝 System Prompt 编辑")
        
        # 1. Prompt 优化对话区 (移至上方)
        with st.expander("💬 AI 优化助手 (点击展开)", expanded=False):
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            chat_container = st.container(height=200)
            for msg in st.session_state.chat_history:
                chat_container.chat_message(msg["role"]).write(msg["content"])

            if prompt_input := st.chat_input("输入修改需求（例如：增加'购买意愿'字段）..."):
                st.session_state.chat_history.append({"role": "user", "content": prompt_input})
                chat_container.chat_message("user").write(prompt_input)
                
                try:
                    client = OpenAI(api_key=st.session_state.api_key, base_url=st.session_state.base_url)
                    optimize_prompt = f"""
                    你是一个 Prompt 优化专家。
                    【当前 Prompt】
                    {st.session_state.system_prompt}
                    【用户需求】
                    {prompt_input}
                    请根据用户需求修改当前 Prompt。只返回修改后的 Prompt 内容，不要包含任何解释。
                    """
                    response = client.chat.completions.create(
                        model=st.session_state.model_name,
                        messages=[{"role": "user", "content": optimize_prompt}],
                        temperature=0.7
                    )
                    ai_reply = response.choices[0].message.content
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
                    chat_container.chat_message("assistant").write(ai_reply)
                    st.info("💡 AI 已生成新 Prompt，请手动复制上方回复内容。")
                except Exception as e:
                    st.error(f"AI 调用失败: {e}")

        # 2. System Prompt 编辑框
        if "system_prompt" not in st.session_state:
            st.session_state.system_prompt = """你是一个专业的数据分类助手。请分析内容，并严格输出以下 JSON 格式：\n{\n    "category": "事件分类",\n    "sentiment": "情感倾向",\n    "reason": "判断理由"\n}"""
        st.session_state.system_prompt = st.text_area("System Prompt", st.session_state.system_prompt, height=400, label_visibility="collapsed")

    with c2:
        st.markdown("##### ⚙️ 测试与优化")
        # 测试控制区 (移至下方，对应右侧布局)
        # 为了平衡布局，我们在右侧上方放一些说明或留白，或者直接放测试控制
        
        st.info("👉 在左侧编辑 Prompt，或使用上方的 AI 助手进行优化。完成后在下方进行测试。")
        
        st.divider()
        
        t1, t2 = st.columns([1, 1])
        with t1:
            test_n = st.number_input("抽样数量", 1, 500, 50)
        with t2:
            st.write("")
            st.write("")
            start_test = st.button("▶️ 开始测试", type="primary", use_container_width=True)

    if start_test:
        sample_df = df.sample(n=min(test_n, len(df)))
        st.session_state.test_indices = sample_df.index.tolist()
        st.session_state.test_results = {}
        
        with st.status("正在进行 AI 测试...", expanded=True) as status:
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                futures = {executor.submit(call_llm, df.loc[i], st.session_state.text_col, st.session_state.context_cols, st.session_state.system_prompt, st.session_state.api_key, st.session_state.base_url, st.session_state.model_name): i for i in sample_df.index}
                for f in concurrent.futures.as_completed(futures):
                    i = futures[f]
                    st.session_state.test_results[i] = parse_json_result(f.result())
            status.update(label="测试完成", state="complete")

    if "test_results" in st.session_state and st.session_state.test_results:
        st.subheader("区域 B: 结果透视")
        results = st.session_state.test_results
        
        # 1. 自动探测所有的 key
        all_keys = set()
        for r in results.values():
            if isinstance(r, dict):
                all_keys.update(r.keys())
        
        # 排除 reason, raw_output 等非标签 key
        candidate_keys = [k for k in all_keys if k not in ["reason", "raw_output", "error"]]
        
        if not candidate_keys:
            st.warning("未检测到有效的 JSON 标签字段，请检查 Prompt 输出格式。")
        else:
            # 让用户选择当前要分析的维度
            if "target_label_key" not in st.session_state:
                st.session_state.target_label_key = candidate_keys[0] if candidate_keys else ""
            
            # 如果之前的 key 不在了，重置
            if st.session_state.target_label_key not in candidate_keys and candidate_keys:
                st.session_state.target_label_key = candidate_keys[0]
                
            c_view1, c_view2 = st.columns([1, 3])
            with c_view1:
                st.session_state.target_label_key = st.selectbox("📊 选择分析维度", candidate_keys, index=candidate_keys.index(st.session_state.target_label_key))
            
            target_key = st.session_state.target_label_key
            
            # 提取当前维度的值
            current_values = [r.get(target_key, "Unknown") for r in results.values() if isinstance(r, dict)]
            
            # 统计展示
            counts = pd.Series(current_values).value_counts()
            st.markdown(f"**📈 '{target_key}' 分布概览**")
            
            # 转置展示
            count_df = counts.to_frame(name="数量").T
            st.dataframe(count_df, use_container_width=True)
        
            unique_cats = sorted(list(set(current_values)))
            if hasattr(st, "pills"):
                selected_cat = st.pills(f"选择 '{target_key}' 查看详情", unique_cats, selection_mode="single")
            else:
                selected_cat = st.radio(f"选择 '{target_key}' 查看详情", unique_cats, horizontal=True)
                
            if selected_cat:
                # 筛选出当前维度符合选定值的 index
                cat_indices = [i for i, r in results.items() if r.get(target_key, "Unknown") == selected_cat]
                display_indices = cat_indices[:5] 
                
                st.write("👀 **显示设置**")
                all_cols = df.columns.tolist()
                default_cols = [st.session_state.text_col] + [c for c in st.session_state.context_cols if c in all_cols]
                show_cols = st.multiselect("选择在详情中展示的原始列", all_cols, default=default_cols)
                
                for idx in display_indices:
                    with st.expander(f"📄 {str(df.loc[idx, st.session_state.text_col])[:50]}...", expanded=True):
                        col_content, col_reason = st.columns([2, 1])
                        with col_content:
                            if show_cols:
                                for col in show_cols:
                                    st.write(f"**{col}:**")
                                    st.caption(df.loc[idx, col])
                            else:
                                st.write("**内容:**")
                                st.write(df.loc[idx, st.session_state.text_col])
                                
                            st.write(f"**AI 判定理由:** {results[idx].get('reason', 'N/A')}")
                            
                            # 显示所有提取到的标签
                            st.write("**AI 提取的所有标签:**")
                            st.json({k: v for k, v in results[idx].items() if k not in ["reason", "raw_output"]})
                            
                            if "raw_output" in results[idx]:
                                st.warning("⚠️ 无法解析 AI 返回的 JSON")
                                st.code(results[idx]["raw_output"], language="json")

                            if st.button("🧠 深度 AI 归因", key=f"btn_reason_{idx}"):
                                reason = get_ai_reasoning(df.loc[idx], st.session_state.text_col, st.session_state.context_cols, f"{target_key}={selected_cat}", st.session_state.api_key, st.session_state.base_url, st.session_state.model_name)
                                st.info(reason)
                        
                        with col_reason:
                            st.write(f"**人工修正 ({target_key}):**")
                            
                            # 获取当前针对该 idx 该 key 的修正值
                            # manual_corrections 结构改为: {idx: {key1: val1, key2: val2}}
                            current_corrections = st.session_state.manual_corrections.get(idx, {})
                            current_fix = current_corrections.get(target_key, selected_cat)
                            
                            options = unique_cats + ["自定义..."]
                            
                            if current_fix in unique_cats:
                                idx_sel = unique_cats.index(current_fix)
                            else:
                                idx_sel = len(unique_cats) # "自定义..."

                            new_fix_select = st.selectbox("修正分类", options, index=idx_sel, key=f"fix_sel_{idx}_{target_key}")
                            
                            final_fix = new_fix_select
                            if new_fix_select == "自定义...":
                                default_custom = current_fix if current_fix not in unique_cats else ""
                                custom_val = st.text_input("输入自定义值", value=default_custom, key=f"fix_custom_{idx}_{target_key}")
                                if custom_val:
                                    final_fix = custom_val
                            
                            # 保存修正逻辑
                            if final_fix != selected_cat:
                                if idx not in st.session_state.manual_corrections:
                                    st.session_state.manual_corrections[idx] = {}
                                st.session_state.manual_corrections[idx][target_key] = final_fix

        st.subheader("区域 C: 优化建议")
        if st.button("根据我的修正生成 Prompt 修改建议"):
            if not st.session_state.manual_corrections:
                st.warning("请先在上方进行一些人工修正。")
            else:
                # 构造传递给 AI 的修正数据，只包含当前关注的 target_key 相关的修正
                target_key = st.session_state.get("target_label_key", "category")
                relevant_corrections = {}
                for idx, corrections in st.session_state.manual_corrections.items():
                    if target_key in corrections:
                        relevant_corrections[idx] = corrections[target_key]
                
                if not relevant_corrections:
                     st.warning(f"请先针对当前分析维度 '{target_key}' 进行一些修正。")
                else:
                    with st.spinner(f"AI 正在分析 '{target_key}' 的误判原因..."):
                        # 临时构造一个只包含 target_key 的 processed_data 视图给分析函数
                        temp_processed_data = {}
                        for idx, res in results.items():
                            temp_processed_data[idx] = {"category": res.get(target_key, "Unknown")} # 欺骗函数名为 category

                        suggestion = analyze_misclassification(
                            relevant_corrections, 
                            temp_processed_data, 
                            df, 
                            st.session_state.text_col, 
                            st.session_state.system_prompt, 
                            st.session_state.api_key, 
                            st.session_state.base_url, 
                            st.session_state.model_name
                        )
                        
                        parts = suggestion.split("【优化后的 System Prompt】")
                        if len(parts) == 2:
                            analysis_text = parts[0].replace("【分析与建议】", "").strip()
                            new_prompt_text = parts[1].strip()
                            
                            st.markdown("#### 💡 分析与建议")
                            st.markdown(analysis_text)
                            
                            st.markdown("#### ✨ 优化后的 System Prompt")
                            st.code(new_prompt_text, language="text")
                            st.info("请复制上面的 Prompt，并替换左侧的 System Prompt 编辑框内容。")
                        else:
                            st.markdown(suggestion)

def render_batch_run(df):
    st.header("3. 全量运行")
    
    st.markdown("#### 🛠️ 确认生产 Prompt")
    st.info("请在此确认最终用于全量跑数的 System Prompt。")
    st.session_state.system_prompt = st.text_area(
        "System Prompt 确认", 
        value=st.session_state.system_prompt, 
        height=300,
        key="batch_prompt_confirm"
    )
    
    if st.button("🚀 运行剩余所有数据"):
        all_indices = df.index.tolist()
        remaining_indices = [i for i in all_indices if i not in st.session_state.processed_data]
        if not remaining_indices:
            st.warning("所有数据已处理。")
        else:
            st.info(f"正在并发处理 {len(remaining_indices)} 条数据...")
            progress_bar = st.progress(0.0)
            status_text_run = st.empty()
            completed = 0
            total = len(remaining_indices)
            with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
                future_to_idx = {
                    executor.submit(call_llm, df.loc[idx], st.session_state.text_col, st.session_state.context_cols, st.session_state.system_prompt, st.session_state.api_key, st.session_state.base_url, st.session_state.model_name): idx 
                    for idx in remaining_indices
                }
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        res = future.result()
                        parsed = parse_json_result(res)
                        st.session_state.processed_data[idx] = parsed
                    except Exception as e:
                        st.session_state.processed_data[idx] = {"error": str(e)}
                    completed += 1
                    progress_bar.progress(completed / total)
                    status_text_run.text(f"进度：{completed}/{total}")
            st.success("全量运行完成！")

    if st.session_state.processed_data:
        st.header("4. 结果质检与验收")
        
        # Merge data
        merged_rows = []
        for idx in df.index:
            base_row = df.loc[idx].to_dict()
            ai_res = st.session_state.processed_data.get(idx, {})
            for k, v in ai_res.items():
                base_row[f"AI_{k}"] = v
            merged_rows.append(base_row)
        final_df = pd.DataFrame(merged_rows)
        
        # QA State Initialization
        if "qa_indices" not in st.session_state:
            st.session_state.qa_indices = []
        if "qa_corrections" not in st.session_state:
            st.session_state.qa_corrections = {}

        # 4.1 Sampling Control
        st.markdown("##### 🔍 抽样质检")
        c_qa_1, c_qa_2, c_qa_3 = st.columns([1, 1, 2])
        with c_qa_1:
            # Detect AI category columns (excluding reason/raw_output)
            ai_cols = [c for c in final_df.columns if c.startswith("AI_") and not any(x in c.lower() for x in ["reason", "raw_output", "error"])]
            
            if not ai_cols:
                st.warning("未找到有效的 AI 标签列。")
                qa_col = None
            else:
                qa_col = st.selectbox("选择质检依据列", ai_cols, key="qa_col_select")
        
        with c_qa_2:
            sample_size = st.number_input("抽样数量", min_value=10, max_value=1000, value=50)
        
        with c_qa_3:
            st.write("")
            st.write("")
            if st.button("🎲 生成新的质检样本", type="primary"):
                try:
                    if len(final_df) > sample_size:
                         qa_sample = final_df.sample(n=sample_size)
                    else:
                        qa_sample = final_df
                    
                    st.session_state.qa_indices = qa_sample.index.tolist()
                    st.session_state.qa_corrections = {} # Reset corrections
                    st.success(f"已抽取 {len(qa_sample)} 条数据进行质检")
                    st.rerun()
                except Exception as e:
                    st.error(f"抽样失败: {e}")

        # 4.2 QA Interface
        if st.session_state.qa_indices and qa_col:
            st.divider()
            qa_df = final_df.loc[st.session_state.qa_indices]
            
            # Error Rate Calculation (based on current qa_col)
            # count how many indices have a correction for the current qa_col
            error_count = 0
            for idx in st.session_state.qa_indices:
                if idx in st.session_state.qa_corrections:
                    if qa_col in st.session_state.qa_corrections[idx]:
                        error_count += 1
            
            total_checked = len(qa_df)
            error_rate = (error_count / total_checked) * 100 if total_checked > 0 else 0
            
            # Metrics Display
            m1, m2, m3 = st.columns(3)
            m1.metric("抽样总量", total_checked)
            m2.metric(f"'{qa_col}' 错误数", error_count, delta_color="inverse")
            m3.metric("当前错误率", f"{error_rate:.1f}%", delta_color="inverse")
            
            # Category Filter
            categories = qa_df[qa_col].astype(str).tolist()
            unique_cats = sorted(list(set(categories)))
            
            if hasattr(st, "pills"):
                selected_cat_qa = st.pills(f"选择 '{qa_col}' 查看样本", unique_cats, selection_mode="single", key="qa_pills")
            else:
                selected_cat_qa = st.radio(f"选择 '{qa_col}' 查看样本", unique_cats, horizontal=True, key="qa_radio")
            
            if selected_cat_qa:
                # Filter indices for this category
                cat_indices = qa_df[qa_df[qa_col] == selected_cat_qa].index.tolist()
                
                if not cat_indices:
                    st.info(f"分类 '{selected_cat_qa}' 下没有样本数据。")
                else:
                    for idx in cat_indices:
                        # Determine current status
                        is_error = False
                        corrected_label = selected_cat_qa
                        
                        if idx in st.session_state.qa_corrections:
                             if qa_col in st.session_state.qa_corrections[idx]:
                                 is_error = True
                                 corrected_label = st.session_state.qa_corrections[idx][qa_col]
                        
                        # Card View
                        with st.expander(f"{'❌' if is_error else '✅'} {str(qa_df.loc[idx, st.session_state.text_col])[:50]}...", expanded=True):
                            qc1, qc2 = st.columns([3, 1])
                            with qc1:
                                st.markdown(f"**内容:** {qa_df.loc[idx, st.session_state.text_col]}")
                                st.caption(f"AI 原判 ({qa_col}): {selected_cat_qa}")
                                if f"AI_reason" in qa_df.columns:
                                     st.caption(f"理由: {qa_df.loc[idx, 'AI_reason']}")
                                
                                # Show other AI tags
                                other_tags = {c: qa_df.loc[idx, c] for c in ai_cols if c != qa_col}
                                if other_tags:
                                    st.caption(f"其他标签: {other_tags}")

                            with qc2:
                                # Manual Correction UI
                                options = unique_cats + ["自定义..."]
                                # Current selection index
                                if corrected_label in unique_cats:
                                    sel_idx = unique_cats.index(corrected_label)
                                else:
                                    sel_idx = len(unique_cats) # Custom
                                    
                                new_correction = st.selectbox("校准分类", options, index=sel_idx, key=f"qa_fix_{idx}_{qa_col}")
                                
                                final_correction = new_correction
                                if new_correction == "自定义...":
                                    default_custom = corrected_label if corrected_label not in unique_cats else ""
                                    final_correction = st.text_input("输入分类", value=default_custom, key=f"qa_custom_{idx}_{qa_col}")
                                
                                # Logic to update state and rerun if changed
                                if final_correction != corrected_label:
                                    if idx not in st.session_state.qa_corrections:
                                        st.session_state.qa_corrections[idx] = {}
                                    
                                    if final_correction != selected_cat_qa:
                                        st.session_state.qa_corrections[idx][qa_col] = final_correction
                                    else:
                                        # Reverted to original
                                        if qa_col in st.session_state.qa_corrections[idx]:
                                            del st.session_state.qa_corrections[idx][qa_col]
                                            # Clean up if empty
                                            if not st.session_state.qa_corrections[idx]:
                                                del st.session_state.qa_corrections[idx]
                                    st.rerun()

        # 4.3 QA Prompt Optimization
        st.divider()
        st.subheader("🤖 AI 质检总结与 Prompt 迭代")
        
        st.info("基于上述质检过程中的人工修正，让 AI 分析误判原因并生成下一次迭代的 System Prompt 建议。")
        
        if st.button("生成质检报告与优化建议", type="primary"):
            if not st.session_state.qa_corrections:
                st.warning("您尚未进行任何人工修正，无法分析误判原因。")
            else:
                with st.spinner("AI 正在分析全量质检结果..."):
                    # Flatten corrections for the current column for analysis
                    flat_corrections = {}
                    current_qa_col = qa_col # Use the currently selected column for analysis
                    
                    for idx, cors in st.session_state.qa_corrections.items():
                        if current_qa_col in cors:
                            flat_corrections[idx] = cors[current_qa_col]
                    
                    if not flat_corrections:
                        st.warning(f"请先针对当前选择的列 '{current_qa_col}' 进行一些修正。")
                    else:
                        suggestion = analyze_misclassification(
                            flat_corrections, 
                            {i: {"category": qa_df.loc[i, current_qa_col]} for i in qa_df.index}, 
                            df, 
                            st.session_state.text_col, 
                            st.session_state.system_prompt, 
                            st.session_state.api_key, 
                            st.session_state.base_url, 
                            st.session_state.model_name
                        )
                        
                        # Display results (same format as Tab 2)
                        parts = suggestion.split("【优化后的 System Prompt】")
                        if len(parts) == 2:
                            analysis_text = parts[0].replace("【分析与建议】", "").strip()
                            new_prompt_text = parts[1].strip()
                            
                            st.markdown("#### 💡 质检分析报告")
                            st.markdown(analysis_text)
                            
                            st.markdown("#### ✨ 下一版 System Prompt 建议")
                            st.code(new_prompt_text, language="text")
                            st.success("您可以复制此 Prompt 用于下一批数据的生产，或在 Prompt 实验室中进一步微调。")
                        else:
                            st.markdown(suggestion)

        st.divider()
        st.subheader("📥 导出最终结果")
        
        # Prepare Export Data
        export_df = final_df.copy()
        
        # Apply corrections
        # qa_corrections format: {idx: {col: val, col2: val2}}
        for idx, col_map in st.session_state.qa_corrections.items():
            for col_name, correct_val in col_map.items():
                # Update the original AI column directly or create a new one?
                # Let's create a Final_ column
                final_col_name = f"Final_{col_name.replace('AI_', '')}"
                export_df.loc[idx, final_col_name] = correct_val
                
                # Also mark as corrected
                export_df.loc[idx, f"Is_Corrected_{col_name}"] = True
                export_df.loc[idx, f"Corrected_From_{col_name}"] = export_df.loc[idx, col_name]

        # Fill Final columns for uncorrected rows
        for col in export_df.columns:
            if col.startswith("AI_") and "reason" not in col and "raw" not in col:
                final_col_name = f"Final_{col.replace('AI_', '')}"
                if final_col_name not in export_df.columns:
                    export_df[final_col_name] = export_df[col]
                else:
                    export_df[final_col_name] = export_df[final_col_name].fillna(export_df[col])

        col_a, col_b = st.columns(2)
        with col_a:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                export_df.to_excel(writer, index=False)
            st.download_button("📥 下载全量结果 (含修正)", buffer.getvalue(), "打标结果_最终.xlsx")
            
        with col_b:
             st.info("导出说明：文件中将包含 'Final_*' 列，为最终采用的分类结果（包含人工修正）。")

# ==========================================
# 主程序逻辑
# ==========================================

def main_app():
    st.title("🤖 AI 数据分类标注工作流（V3 稳定版）")
    
    for key, default in [("df", None), ("processed_data", {}), ("manual_corrections", {}), ("test_results", {}), ("error_flags", {}), ("qa_indices", []), ("qa_corrections", {})]:
        if key not in st.session_state: st.session_state[key] = default

    # 增加自定义 CSS 放大 Tab 字体
    st.markdown("""
    <style>
        div[data-baseweb="tab-list"] p {
            font-size: 1.2rem;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["1️⃣ 第一步：数据上传", "2️⃣ 第二步：Prompt 实验室", "3️⃣ 第三步：批量生产"])
    
    with tab1:
        st.header("1. 配置与上传")
        with st.expander("📂 文件与 API 设置", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                uploaded_file = st.file_uploader("上传 Excel 文件", type=["xlsx", "xls"])
                if uploaded_file:
                    sheet_names = load_excel_sheets(uploaded_file)
                    if sheet_names:
                        selected_sheet = st.selectbox("选择 Sheet", sheet_names)
                        if st.button("确认加载数据"):
                             with st.spinner("正在读取 Excel..."):
                                df_loaded = load_excel(uploaded_file, selected_sheet)
                                if df_loaded is not None:
                                    st.session_state.df = df_loaded.reset_index(drop=True)
                                    st.success("数据已就绪，请切换到【Prompt实验室】进行调试")
                if st.session_state.df is not None:
                    st.info(f"当前已加载: {len(st.session_state.df)} 条")
            with col2:
                if "api_key" not in st.session_state: st.session_state.api_key = "sk-xxxx"
                st.session_state.api_key = st.text_input("API Key", value=st.session_state.api_key, type="password")
                if "base_url" not in st.session_state: st.session_state.base_url = "https://ark.cn-beijing.volces.com/api/v3"
                st.session_state.base_url = st.text_input("Base URL", value=st.session_state.base_url)
                if "model_name" not in st.session_state: st.session_state.model_name = "ep-20250530201032-d9f2d"
                st.session_state.model_name = st.text_input("Model Name", value=st.session_state.model_name)

        if st.session_state.df is not None:
            c1, c2 = st.columns(2)
            st.session_state.text_col = c1.selectbox("选择主要内容列", st.session_state.df.columns)
            st.session_state.context_cols = c2.multiselect("选择辅助信息列", [c for c in st.session_state.df.columns if c != st.session_state.text_col])
    
    with tab2:
        if st.session_state.df is None:
            st.warning("请先在 Tab 1 上传数据")
        else:
            render_prompt_lab(st.session_state.df)
    
    with tab3:
        if st.session_state.df is None:
            st.warning("请先上传数据")
        else:
            render_batch_run(st.session_state.df)

def main_app_old():
    # 清除启动提示
    status_text.empty()
    st.title("🤖 AI 数据分类标注工作流（V3 稳定版）")

    if "df" not in st.session_state:
        st.session_state.df = None
    if "sample_indices" not in st.session_state:
        st.session_state.sample_indices = []
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = {}
    if "error_flags" not in st.session_state:
        st.session_state.error_flags = {}

    st.header("1. 配置与上传")

    with st.expander("📂 文件与 API 设置", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            uploaded_file = st.file_uploader("上传 Excel 文件", type=["xlsx", "xls"])
            if uploaded_file:
                # 1. 先读取 sheet 列表
                sheet_names = load_excel_sheets(uploaded_file)
                if sheet_names:
                    selected_sheet = st.selectbox("选择 Sheet", sheet_names)
                    
                    # 2. 只有当用户选择了 sheet 且尚未加载数据时才读取
                    if st.button("确认加载数据"):
                         with st.spinner("正在读取 Excel..."):
                            df_loaded = load_excel(uploaded_file, selected_sheet)
                            if df_loaded is not None:
                                df_loaded = df_loaded.reset_index(drop=True)
                                st.session_state.df = df_loaded
                                st.success(f"已加载 {len(df_loaded)} 条数据")
                
            if st.session_state.df is not None:
                st.info(f"当前已加载: {len(st.session_state.df)} 条")

        with col2:
            api_key = st.text_input(
                "API Key",
                value="sk-xxxx",
                type="password",
            )
            base_url = st.text_input("Base URL", value="https://ark.cn-beijing.volces.com/api/v3")
            model_name = st.text_input("Model Name", value="ep-20250530201032-d9f2d")

    if st.session_state.df is None:
        st.info("请先上传 Excel 文件。")
        return

    df = st.session_state.df

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        text_col = st.selectbox("选择主要内容列", df.columns)
    with col_sel2:
        context_cols = st.multiselect("选择辅助信息列", [c for c in df.columns if c != text_col])

    st.subheader("🤖 Prompt 优化助手")
    
    col_prompt, col_chat = st.columns([1, 1])
    
    with col_prompt:
        default_prompt = """你是一个专业的数据分类助手。请分析内容，并严格输出以下 JSON 格式：
{
    "category": "事件分类",
    "sentiment": "情感倾向",
    "reason": "判断理由"
}"""
        system_prompt = st.text_area("System Prompt (可直接修改)", value=default_prompt, height=300)

    with col_chat:
        st.write("💬 **与 AI 对话优化 Prompt**")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # 显示历史对话
        chat_container = st.container(height=200)
        for msg in st.session_state.chat_history:
            chat_container.chat_message(msg["role"]).write(msg["content"])

        if user_input := st.chat_input("输入修改意见（例如：帮我增加一个'购买意愿'字段）..."):
            # 用户消息上屏
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            chat_container.chat_message("user").write(user_input)

            # 调用 AI 优化 Prompt
            try:
                client = OpenAI(api_key=api_key, base_url=base_url)
                optimize_prompt = f"""
                你是一个 Prompt 优化专家。
                
                【当前 Prompt】
                {system_prompt}
                
                【用户需求】
                {user_input}
                
                请根据用户需求修改当前 Prompt。只返回修改后的 Prompt 内容，不要包含任何解释或其他废话。
                确保保留 JSON 输出格式的要求。
                """
                
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": optimize_prompt}],
                    temperature=0.7
                )
                
                ai_reply = response.choices[0].message.content
                
                # AI 回复上屏
                st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
                chat_container.chat_message("assistant").write(ai_reply)
                
                # 提示用户手动复制
                st.info("💡 AI 已生成新 Prompt，请手动复制上面的回复内容并粘贴到左侧输入框中。")
                
            except Exception as e:
                st.error(f"AI 调用失败: {e}")

    st.header("2. 测试运行")
    col_test1, col_test2 = st.columns([1, 3])
    with col_test1:
        test_n = st.number_input("设置测试条数", min_value=1, max_value=1000, value=50, step=10)
    
    with col_test2:
        # 添加一些垂直间距，让按钮和输入框对齐
        st.write("") 
        st.write("")
        if st.button(f"🧪 测试 {test_n} 条"):
            sample_n = min(test_n, len(df))
            sample_df = df.sample(n=sample_n, random_state=42)
            st.session_state.sample_indices = sample_df.index.tolist()

            st.info("正在测试...")
            progress_bar = st.progress(0.0)

            # 改为并发执行测试
            with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
                future_to_idx = {
                    executor.submit(call_llm, row, text_col, context_cols, system_prompt, api_key, base_url, model_name): idx 
                    for idx, row in sample_df.iterrows()
                }
                
                completed = 0
                total = len(sample_df)
                
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        res = future.result()
                        parsed = parse_json_result(res)
                        st.session_state.processed_data[idx] = parsed
                    except Exception as e:
                        st.session_state.processed_data[idx] = {"error": str(e)}
                    
                    completed += 1
                    progress_bar.progress(completed / total)

            st.success("测试完成！")

    if st.session_state.sample_indices:
        st.subheader("📝 校对结果")
        st.caption("请在下方表格中直接修改数据。如果 AI 结果有误，请勾选『❌ 标记错误』列。")

        sample_rows = []
        for idx in st.session_state.sample_indices:
            base_row = df.loc[idx].to_dict()
            ai_res = st.session_state.processed_data.get(idx, {})
            for k, v in ai_res.items():
                base_row[f"AI_{k}"] = v
            
            # 添加标记错误列
            base_row["❌ 标记错误"] = st.session_state.error_flags.get(idx, False)
            base_row["_original_index"] = idx
            sample_rows.append(base_row)

        # 配置列显示，让 Checkbox 更直观
        column_config = {
            "❌ 标记错误": st.column_config.CheckboxColumn(
                "标记错误",
                help="如果 AI 结果不对，请勾选此项",
                default=False,
            )
        }

        sample_edit_df = st.data_editor(
            pd.DataFrame(sample_rows), 
            num_rows="dynamic",
            column_config=column_config
        )
        
        if not sample_edit_df.empty:
            error_count = 0
            for _, r in sample_edit_df.iterrows():
                idx = int(r["_original_index"])
                
                # 更新 AI 结果数据
                updated_res = {k.replace("AI_", ""): v for k, v in r.items() if isinstance(k, str) and k.startswith("AI_")}
                st.session_state.processed_data[idx] = updated_res

                # 更新错误标记状态
                is_err = r.get("❌ 标记错误", False)
                st.session_state.error_flags[idx] = is_err
                if is_err:
                    error_count += 1
            
            # 实时显示错误率指标
            if len(sample_rows) > 0:
                error_rate = error_count / len(sample_rows)
                st.metric("🚩 当前错误率", f"{error_rate:.1%}", f"已标记 {error_count} 条错误")

    st.header("3. 全量运行")
    if st.button("🚀 运行剩余所有数据"):
        all_indices = df.index.tolist()
        remaining_indices = [i for i in all_indices if i not in st.session_state.processed_data]
        
        if not remaining_indices:
            st.warning("所有数据已处理。")
        else:
            st.info(f"正在并发处理 {len(remaining_indices)} 条数据...")
            progress_bar = st.progress(0.0)
            status_text_run = st.empty()
            
            completed = 0
            total = len(remaining_indices)
            
            # 增加并发数到 50
            with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
                future_to_idx = {
                    executor.submit(call_llm, df.loc[idx], text_col, context_cols, system_prompt, api_key, base_url, model_name): idx 
                    for idx in remaining_indices
                }
                
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        res = future.result()
                        parsed = parse_json_result(res)
                        st.session_state.processed_data[idx] = parsed
                    except Exception as e:
                        st.session_state.processed_data[idx] = {"error": str(e)}
                    
                    completed += 1
                    progress_bar.progress(completed / total)
                    status_text_run.text(f"进度：{completed}/{total}")
            
            st.success("全量运行完成！")

    if st.session_state.processed_data:
        st.header("4. 导出与质检")
        
        merged_rows = []
        for idx in df.index:
            base_row = df.loc[idx].to_dict()
            ai_res = st.session_state.processed_data.get(idx, {})
            for k, v in ai_res.items():
                base_row[f"AI_{k}"] = v
            merged_rows.append(base_row)
        
        final_df = pd.DataFrame(merged_rows)
        
        col_a, col_b = st.columns(2)
        with col_a:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                final_df.to_excel(writer, index=False)
            st.download_button("📥 下载全量结果", buffer.getvalue(), "打标结果_全量.xlsx")
            
        with col_b:
            ai_cols = [c for c in final_df.columns if c.startswith("AI_")]
            if ai_cols:
                qa_col = st.selectbox("选择抽样列", ai_cols)
                if st.button("生成抽样表"):
                    qa_df = stratified_sample(final_df, qa_col, frac=0.1)
                    buffer_qa = io.BytesIO()
                    with pd.ExcelWriter(buffer_qa, engine="openpyxl") as writer:
                        qa_df.to_excel(writer, index=False)
                    st.download_button("📥 下载质检抽样表", buffer_qa.getvalue(), f"质检_{qa_col}.xlsx")

if __name__ == "__main__":
    main_app()
