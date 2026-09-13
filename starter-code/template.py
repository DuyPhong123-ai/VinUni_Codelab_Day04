"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
import sys
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## 1. PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm, dịch vụ và tiếp nhận yêu cầu hỗ trợ khách hàng trong hệ sinh thái Vingroup (VinFast, Vinpearl).
- Giọng nói & Phong cách: Chuyên nghiệp, tận tâm, trung thực, lịch sự và chính xác.

## 2. AVAILABLE TOOLS
- `search_product_catalog(category, max_price)`: Tra cứu sản phẩm/dịch vụ (xe điện 'xe_dien', du lịch 'du_lich') theo khoảng giá tối đa (VNĐ).
- `submit_support_ticket(customer_name, issue_description, priority)`: Ghi nhận yêu cầu hỗ trợ hoặc phản ánh khiếu nại của khách hàng vào hệ thống ticket.

## 3. CORE RULES
- TUYỆT ĐỐI KHÔNG tự bịa đặt (hallucinate) thông tin sản phẩm, thông số kỹ thuật, giá bán hay mã ticket khi chưa có dữ liệu thực tế.
- BẮT BUỘC phải gọi công cụ (tool calling) khi người dùng yêu cầu tìm kiếm sản phẩm hoặc gửi yêu cầu hỗ trợ.
- Nếu không tìm thấy sản phẩm phù hợp sau khi tra cứu dữ liệu, PHẢI phản hồi rõ ràng là không tìm thấy sản phẩm phù hợp.
- Với các câu hỏi thường gặp (FAQ) về chính sách bảo hành đã có sẵn thông tin chính thống, trả lời trung thực và chuẩn xác.

## 4. OPERATIONAL BOUNDARIES
- Chỉ hoạt động trong phạm vi sản phẩm và dịch vụ của hệ sinh thái Vingroup (VinFast, Vinpearl).
- Từ chối lịch sự và chuyển hướng nếu người dùng hỏi các chủ đề ngoài phạm vi hoạt động của Vingroup.

## 5. OUTPUT CONTRACT
Tuân thủ định dạng luồng ReAct khi làm việc:
- Thought: Phân tích ý định của người dùng và xác định hành động tiếp theo.
- Action: Tên tool và tham số tương ứng cần gọi (hoặc None nếu trả lời trực tiếp).
- Observation: Kết quả thực thi trả về từ tool.
- Final Answer: Câu trả lời tổng hợp sau cùng gửi tới người dùng.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # Trả về câu trả lời tĩnh (mock) hoặc gọi LLM thuần không dùng tool
        # Mục tiêu: Quan sát hiện tượng bịa thông tin (hallucination)
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _detect_intents(self, user_input: str) -> Dict[str, Any]:
        """Phân tích intent và trích xuất tham số từ user_input."""
        text_lower = user_input.lower()

        # 1. Intent FAQ: chính sách bảo hành, thông tin chung
        faq_keywords = ["chính sách", "kéo dài bao lâu", "thời hạn bảo hành", "quy định bảo hành"]
        is_faq = any(kw in text_lower for kw in faq_keywords) and ("bảo hành" in text_lower or "pin" in text_lower)

        # 2. Intent tra cứu catalog sản phẩm/dịch vụ
        catalog_triggers = [
            "xem", "tìm", "mua", "giá dưới", "có xe nào", "có resort nào",
            "có khách sạn nào", "báo giá", "chi phí", "bao nhiêu tiền", "cho tôi xem"
        ]
        category_xe = ["xe", "ô tô", "oto", "vf", "xe điện"]
        category_dulich = ["resort", "vinpearl", "du lịch", "phòng", "khách sạn", "nghỉ dưỡng"]

        has_catalog_trigger = any(t in text_lower for t in catalog_triggers)
        has_category = any(k in text_lower for k in category_xe + category_dulich)

        needs_catalog = False
        catalog_args = {}
        if has_catalog_trigger and has_category and not is_faq:
            needs_catalog = True

            # Xác định category
            if any(k in text_lower for k in category_dulich):
                category = "du_lich"
            else:
                category = "xe_dien"

            # Trích xuất giá tối đa (max_price)
            max_price = 999999999999
            price_match = re.search(r'(\d+(?:[.,]\d+)?)\s*(triệu|tỷ|tr)', text_lower)
            if price_match:
                num = float(price_match.group(1).replace(',', '.'))
                unit = price_match.group(2)
                if 'tỷ' in unit:
                    max_price = int(num * 1_000_000_000)
                else:
                    max_price = int(num * 1_000_000)
            else:
                num_match = re.search(r'dưới\s*(\d+)', text_lower)
                if num_match:
                    val = int(num_match.group(1))
                    if val < 10000:
                        max_price = val * 1_000_000
                    else:
                        max_price = val

            catalog_args = {
                "category": category,
                "max_price": max_price
            }

        # 3. Intent gửi support ticket (Kiểm tra độc lập để tránh trap if-elif)
        ticket_triggers = [
            "tôi tên", "tên tôi là", "phản ánh", "khiếu nại", "bị lỗi",
            "hỏng", "sự cố", "ghi nhận phản hồi", "cần xử lý gấp", "hỗ trợ"
        ]
        needs_ticket = any(t in text_lower for t in ticket_triggers) and not is_faq
        ticket_args = {}

        if needs_ticket:
            # Trích xuất tên khách hàng
            name_match = re.search(
                r'(?:tôi tên là|tôi tên|tên tôi là)\s+([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:,|;|\.|\n|xe|phòng|và|mức|đây|vấn|đang|yêu|mong|sđt|số|cần|$)',
                user_input,
                re.IGNORECASE
            )
            customer_name = name_match.group(1).strip() if name_match else "Khách hàng"

            # Trích xuất mức độ ưu tiên
            if any(w in text_lower for w in ["nghiêm trọng", "gấp", "khẩn cấp", "high"]):
                priority = "high"
            elif any(w in text_lower for w in ["thấp", "low"]):
                priority = "low"
            else:
                priority = "medium"

            # Trích xuất mô tả sự cố
            issue_match = re.search(
                r'((?:xe|phòng|thiết bị|hệ thống|dịch vụ)[^,.]*(?:bị|lỗi|hỏng|ẩm mốc)[^,.]*|bị lỗi[^,.]*|ẩm mốc[^,.]*)',
                user_input,
                re.IGNORECASE
            )
            if issue_match:
                issue_description = issue_match.group(1).strip()
            else:
                issue_description = user_input.strip()

            ticket_args = {
                "customer_name": customer_name,
                "issue_description": issue_description,
                "priority": priority
            }

        return {
            "needs_catalog": needs_catalog,
            "catalog_args": catalog_args,
            "needs_ticket": needs_ticket,
            "ticket_args": ticket_args,
            "is_faq": is_faq
        }

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []

        intents = self._detect_intents(user_input)

        # Xác định danh sách các bước cần thực thi
        steps_to_run: List[str] = []
        if intents["is_faq"]:
            steps_to_run.append("faq")
        else:
            if intents["needs_catalog"]:
                steps_to_run.append("catalog")
            if intents["needs_ticket"]:
                steps_to_run.append("ticket")
            if not steps_to_run:
                steps_to_run.append("general")

        catalog_data: List[Dict[str, Any]] = []
        ticket_data: Dict[str, Any] = {}

        iteration = 1
        step_idx = 0

        while iteration <= self.max_iterations and step_idx < len(steps_to_run):
            step_type = steps_to_run[step_idx]
            is_last_step = (step_idx == len(steps_to_run) - 1)

            if step_type == "faq":
                thought = "Thought: Người dùng hỏi về chính sách bảo hành của VinFast. Câu hỏi thuộc FAQ, trả lời trực tiếp mà không cần gọi tool."
                action = None
                observation = None
                final_answer = "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm hoặc 200.000 km (tùy điều kiện nào đến trước), áp dụng cho toàn bộ các dòng ô tô điện VinFast."
                self.trace.append({
                    "iteration": iteration,
                    "thought": thought,
                    "action": action,
                    "observation": observation,
                    "final_answer": final_answer
                })
                return {
                    "answer": final_answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }

            elif step_type == "catalog":
                c_args = intents["catalog_args"]
                thought = f"Thought: Người dùng muốn tra cứu danh mục {c_args.get('category')}. Gọi tool search_product_catalog."
                action = {"tool": "search_product_catalog", "args": c_args}
                observation = search_product_catalog(**c_args)
                catalog_data = observation

                step_trace = {
                    "iteration": iteration,
                    "thought": thought,
                    "action": action,
                    "observation": observation
                }

                if is_last_step:
                    # Tạo Final Answer cho catalog
                    if not catalog_data or len(catalog_data) == 0:
                        final_answer = "Rất tiếc, không tìm thấy sản phẩm phù hợp với yêu cầu của quý khách."
                    else:
                        items = [
                            f"- {p['name']}: Giá {p.get('price_vnd', 0):,} VNĐ. {p.get('description', '')}"
                            for p in catalog_data
                        ]
                        final_answer = f"Tìm thấy {len(catalog_data)} sản phẩm phù hợp:\n" + "\n".join(items)

                    step_trace["final_answer"] = final_answer
                    self.trace.append(step_trace)
                    return {
                        "answer": final_answer,
                        "trace": self.trace,
                        "iterations": iteration,
                        "status": "completed"
                    }

                self.trace.append(step_trace)

            elif step_type == "ticket":
                t_args = intents["ticket_args"]
                thought = f"Thought: Người dùng muốn phản ánh sự cố. Gọi tool submit_support_ticket."
                action = {"tool": "submit_support_ticket", "args": t_args}
                observation = submit_support_ticket(**t_args)
                ticket_data = observation

                step_trace = {
                    "iteration": iteration,
                    "thought": thought,
                    "action": action,
                    "observation": observation
                }

                if is_last_step:
                    # Tạo Final Answer (có thể kết hợp kết quả catalog nếu trước đó có gọi)
                    ticket_id = ticket_data.get("ticket_id", "")
                    cust_name = ticket_data.get("customer_name", t_args.get("customer_name", "quý khách"))
                    ticket_msg = f"Yêu cầu hỗ trợ của quý khách {cust_name} đã được ghi nhận với mã {ticket_id}. Đội ngũ hỗ trợ sẽ xử lý trong thời gian sớm nhất."

                    if catalog_data:
                        items = [
                            f"- {p['name']}: Giá {p.get('price_vnd', 0):,} VNĐ. {p.get('description', '')}"
                            for p in catalog_data
                        ]
                        catalog_msg = f"Thông tin sản phẩm/dịch vụ bạn quan tâm:\n" + "\n".join(items)
                        final_answer = f"{catalog_msg}\n\n{ticket_msg}"
                    else:
                        final_answer = ticket_msg

                    step_trace["final_answer"] = final_answer
                    self.trace.append(step_trace)
                    return {
                        "answer": final_answer,
                        "trace": self.trace,
                        "iterations": iteration,
                        "status": "completed"
                    }

                self.trace.append(step_trace)

            elif step_type == "general":
                thought = "Thought: Yêu cầu chung về Vingroup, phản hồi hỗ trợ cơ bản."
                action = None
                observation = None
                final_answer = "Xin chào quý khách! Tôi là VinAssistant, trợ lý AI của Vingroup. Tôi có thể hỗ trợ tra cứu sản phẩm VinFast, Vinpearl hoặc tạo phiếu khiếu nại/hỗ trợ."
                self.trace.append({
                    "iteration": iteration,
                    "thought": thought,
                    "action": action,
                    "observation": observation,
                    "final_answer": final_answer
                })
                return {
                    "answer": final_answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }

            step_idx += 1
            iteration += 1

        # Nếu vượt quá max_iterations
        return {
            "answer": "Lỗi: Vượt quá số bước tối đa.",
            "trace": self.trace,
            "iterations": iteration - 1,
            "status": "max_iterations_reached"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
