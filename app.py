from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import json
from langfuse import Langfuse
from openai import OpenAI
from datetime import datetime
from pathlib import Path
import uuid
from typing import List, Dict, Optional
from enum import StrEnum, auto
from typing import Union

import sys
from clarifications_structure import NeedMoreClarifications, FollowUps

openai_client = OpenAI()
langfuse_client = Langfuse()

class MessageType(StrEnum):
    TOOL_MESSAGE = auto()
    USER_MESSAGE = auto()
    SYSTEM_MESSAGE = auto()

class PromptMode(StrEnum):
    CONCISE = auto()
    DETAILED = auto()

# Configure the page
st.set_page_config(
    page_title="Clarification Agent",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ChatManager:
    def __init__(self, chats_dir: str = "streamlit/chat_storage"):
        self.chats_dir = Path(chats_dir)
        self.chats_dir.mkdir(exist_ok=True)
        self.index_file = self.chats_dir / "chat_index.json"
    
    def get_chat_list(self) -> List[Dict]:
        """Get list of all chats from index file"""
        if not self.index_file.exists():
            return []
        
        try:
            with open(self.index_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def create_new_chat(self, title: str = None) -> str:
        """Create a new chat and return its ID"""
        chat_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Create chat file
        chat_file = self.chats_dir / f"{chat_id}.jsonl"
        chat_file.touch()
        
        # Update index
        chat_index = self.get_chat_list()
        chat_info = {
            "id": chat_id,
            "title": title or f"Chat {len(chat_index) + 1}",
            "created_at": timestamp,
            "last_updated": timestamp,
            "message_count": 0
        }
        chat_index.append(chat_info)
        
        with open(self.index_file, 'w') as f:
            json.dump(chat_index, f, indent=2)
        
        return chat_id
    
    def update_chat_title(self, chat_id: str, title: str):
        """Update chat title in index"""
        chat_index = self.get_chat_list()
        for chat in chat_index:
            if chat["id"] == chat_id:
                chat["title"] = title
                break
        
        with open(self.index_file, 'w') as f:
            json.dump(chat_index, f, indent=2)
    
    def update_chat_toggle(self, chat_id: str, toggle_state: int):
        """Update chat toggle state in index"""
        chat_index = self.get_chat_list()
        for chat in chat_index:
            if chat["id"] == chat_id:
                chat["toggle_state"] = toggle_state
                break
        
        with open(self.index_file, 'w') as f:
            json.dump(chat_index, f, indent=2)
    
    def get_chat_toggle(self, chat_id: str) -> int:
        """Get chat toggle state"""
        chat_index = self.get_chat_list()
        for chat in chat_index:
            if chat["id"] == chat_id:
                return chat.get("toggle_state", 1)  # Default to 1 if not found
        return 1
    
    @property
    def prompt_mode(self) -> PromptMode:
        toggle_mode = st.session_state.chat_manager.get_chat_toggle(st.session_state.current_chat_id)
        return PromptMode.CONCISE if toggle_mode else PromptMode.DETAILED

    def add_message(self, chat_id: str, message: Dict):
        """Append a message to chat file"""
        chat_file = self.chats_dir / f"{chat_id}.jsonl"
        
        # Add timestamp to message
        message["timestamp"] = datetime.now().isoformat()
        
        # Ensure message has all required fields (with defaults)
        message_to_save = {
            "role": message.get("role"),
            "content": message.get("content"),
            "timestamp": message["timestamp"],
            "type": message["type"]
        }
        
        # Append to JSONL file
        with open(chat_file, 'a') as f:
            f.write(json.dumps(message_to_save) + '\n')
        
        # Update index with last_updated and message count
        chat_index = self.get_chat_list()
        for chat in chat_index:
            if chat["id"] == chat_id:
                chat["last_updated"] = message["timestamp"]
                chat["message_count"] = chat.get("message_count", 0) + 1
                break
        
        with open(self.index_file, 'w') as f:
            json.dump(chat_index, f, indent=2)
    
    def get_chat_messages(self, chat_id: str) -> List[Dict]:
        """Get all messages from a chat"""
        chat_file = self.chats_dir / f"{chat_id}.jsonl"
        
        if not chat_file.exists():
            return []
        
        messages = []
        with open(chat_file, 'r') as f:
            for line in f:
                if line.strip():
                    messages.append(json.loads(line))
        
        return messages
    
    def delete_chat(self, chat_id: str):
        """Delete a chat and remove from index"""
        # Delete chat file
        chat_file = self.chats_dir / f"{chat_id}.jsonl"
        if chat_file.exists():
            chat_file.unlink()
        
        # Remove from index
        chat_index = self.get_chat_list()
        chat_index = [chat for chat in chat_index if chat["id"] != chat_id]
        
        with open(self.index_file, 'w') as f:
            json.dump(chat_index, f, indent=2)

class ChatAssistant:
    def __init__(self):
        pass
    
    def find_unfinished_conversation(self, messages: List[Dict]) -> Optional[List[Dict]]:
        """
        Find unfinished conversation thread if last message is from user
        Returns list of consecutive user/tool messages leading up to the unanswered user message
        """
        if not messages:
            return []
        
        # Check if last message is from user (unanswered)
        last_message = messages[-1]
        if last_message.get("role") != "user":
            return []
        
        # Work backwards from the end to find the conversation thread
        unfinished_thread = []

        # breakpoint()
        
        for i in range(len(messages) - 1, -1, -1):
            current_message = messages[i]
            role = current_message["role"]
            message_type = current_message['type']  # Default to "regular" if no type
            
            # Include user messages and tool messages
            if role == "user" or (role == "assistant" and message_type == "tool"):
                unfinished_thread.append(current_message)
            elif role == "assistant" and message_type != "tool":
                # Stop when we hit a regular assistant message
                break
        
        unfinished_thread.reverse()  # Reverse to maintain original order
        return unfinished_thread if unfinished_thread else []

    def needs_clarification(self, context: List[Dict], promptMode: PromptMode) -> Union[bool, str]:
        """
        Determine if the user input needs clarification
        Returns (needs_clarification, clarification_question)
        """
        past_questions: list[str] = []
        for conversation_message in context:
            if conversation_message.get("role") == "user":
                try:
                    past_questions[-1] += f"Answers: {conversation_message.get('content', '')}"
                except IndexError:
                    past_questions.append(f"Answers: {conversation_message.get('content', '')}")
            elif conversation_message.get("role") == "assistant":
                past_questions.append(f"Question: {conversation_message.get('content', '')}\n")
        print(past_questions)
        print(context)

        need_more_clarifications_prompt = "canvas/clarifier/needs-more-clarifications" if promptMode == PromptMode.DETAILED else "canvas/clarifier/needs-more-clarifications-concise"
        clarifier_system_prompt = "canvas/clarifier/clarifierSystemPrompt" if promptMode == PromptMode.DETAILED else "canvas/clarifier/ClarifierSystemPromptConcise" 

        response = openai_client.responses.parse(
            model="gpt-4.1-mini",
            input=[
                {'role': 'system', 'content': langfuse_client.get_prompt(need_more_clarifications_prompt).get_langchain_prompt()},
                {'role': 'user', 'content': f"Here are the follow-ups and answers: {"\n\n".join(past_questions)}"}
            ],
            text_format=NeedMoreClarifications
        )

        need_more_clarifications = response.output_parsed

        assert need_more_clarifications is not None, "NeedMoreClarifications response cannot be None"

        if need_more_clarifications.need_more is False:
            return False
        
        response = openai_client.responses.parse(
            model="gpt-4.1",
            input=[
                {"role": "system", "content":langfuse_client.get_prompt(clarifier_system_prompt).get_langchain_prompt()
                },
                {"role": "user", "content":
                    f"Original question: {context[0]['content']}\n"
                    f"Suggested follow-up questions:\n{str(need_more_clarifications.need_more.more_follow_ups)}"
                    f"Previous clarifications:\n{"\n\n".join(past_questions)}"
                }
            ],
            text_format=FollowUps
        )
        follow_up = response.output_parsed
        assert isinstance(follow_up, FollowUps), "Response should be of type FollowUps"

        questions = ""
        for question in follow_up.more_follow_ups:
            if question.strip():
                questions += f"{question.strip()}\n"
        
        return questions.strip()

    def process_unfinished_conversation(self, unfinished_thread: List[Dict], prompt_mode: PromptMode) -> Union[bool, str]:
        """
        Process the unfinished conversation thread
        This is where you'd implement your specific processing logic
        """
        # Example processing - replace with your actual logic

        return self.needs_clarification(unfinished_thread, prompt_mode)

# Initialize session state
if "chat_manager" not in st.session_state:
    st.session_state.chat_manager = ChatManager()

if "assistant" not in st.session_state:
    st.session_state.assistant = ChatAssistant()

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for chat management
with st.sidebar:
    st.title("💬 Chats")
    
    # New chat button
    if st.button("➕ New Chat", type="primary", use_container_width=True):
        chat_id = st.session_state.chat_manager.create_new_chat()
        st.session_state.current_chat_id = chat_id
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # List existing chats
    chat_list = st.session_state.chat_manager.get_chat_list()
    chat_list.sort(key=lambda x: x["last_updated"], reverse=True)
    
    for chat in chat_list:
        col1, col2 = st.columns([4, 1])
        
        with col1:
            if st.button(
                f"💭 {chat['title'][:25]}{'...' if len(chat['title']) > 25 else ''}",
                key=f"chat_{chat['id']}",
                use_container_width=True,
                type="secondary" if chat['id'] != st.session_state.current_chat_id else "primary"
            ):
                st.session_state.current_chat_id = chat['id']
                st.session_state.messages = st.session_state.chat_manager.get_chat_messages(chat['id'])
                
                # Check for unfinished conversation when switching to a chat
                unfinished_thread = st.session_state.assistant.find_unfinished_conversation(st.session_state.messages)

                if len(unfinished_thread) > 0:
                    # Process the unfinished conversation automatically
                    thread_response = st.session_state.assistant.process_unfinished_conversation(unfinished_thread, st.session_state.chat_manager.prompt_mode)
                    # Processing unfinished conversation
                    if thread_response is False:
                        assistant_message = {
                            "role": "assistant",
                            "content": "Fin.",
                            "type": MessageType.SYSTEM_MESSAGE
                        }
                    else:
                        assistant_message = {
                            "role": "assistant",
                            "content": thread_response,
                            "type": MessageType.TOOL_MESSAGE
                        }
                    st.session_state.messages.append(assistant_message)
                    st.session_state.chat_manager.add_message(chat['id'], assistant_message)
                else:
                    print("No unfinished conversation found, ready for new input.")
                    print(unfinished_thread)

                st.rerun()
        
        with col2:
            if st.button("🗑️", key=f"delete_{chat['id']}", help="Delete chat"):
                st.session_state.chat_manager.delete_chat(chat['id'])
                if st.session_state.current_chat_id == chat['id']:
                    st.session_state.current_chat_id = None
                    st.session_state.messages = []
                st.rerun()

# Main chat interface
st.title("💬 Chat Assistant")

# Show current chat or prompt to create one
if st.session_state.current_chat_id is None:
    st.info("👈 Select an existing chat or create a new one to get started!")
else:
    # Display chat messages
    for message in st.session_state.messages:
        role = message["role"]
        message_type = message["type"]
        content = message["content"]
        
        if role == "user":
            with st.chat_message("user"):
                st.write(content)
        elif role == "assistant":
            if message_type == "tool":
                with st.chat_message("assistant", avatar="🔧"):
                    st.write(f"🔧 **Tool Message:** {content}")
            else:
                with st.chat_message("assistant"):
                    st.write(content)
    
    if st.session_state.current_chat_id:
        current_toggle = st.session_state.chat_manager.get_chat_toggle(st.session_state.current_chat_id)
        
        st.divider()
        new_toggle = st.toggle(
            "Concise Questions",
            value=bool(current_toggle),
            key=f"current_toggle_{st.session_state.current_chat_id}",
            help="Enable the concise option if you want the model to ask as few questions as possible"
        )
        
        # Update toggle state if changed
        new_toggle_value = 1 if new_toggle else 0
        if new_toggle_value != current_toggle:
            st.session_state.chat_manager.update_chat_toggle(st.session_state.current_chat_id, new_toggle_value)
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat
        user_message = {"role": "user", "content": prompt, 'type': MessageType.USER_MESSAGE}
        st.session_state.messages.append(user_message)
        st.session_state.chat_manager.add_message(st.session_state.current_chat_id, user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        relevant_chat_history = st.session_state.assistant.find_unfinished_conversation(st.session_state.messages)
        


        needs_clarification = st.session_state.assistant.needs_clarification(relevant_chat_history, st.session_state.chat_manager.prompt_mode)
        assistant_response = {'role':'assistant'}
        
        if needs_clarification is False:
            # Generate clarification question
            assistant_response['content'] = 'Fin.'
            assistant_response['type'] = MessageType.SYSTEM_MESSAGE
        else:
            # Generate actual response
            assistant_response['content'] = needs_clarification
            assistant_response['type'] = MessageType.TOOL_MESSAGE
        
        # Add message to chat
        st.session_state.messages.append(assistant_response)
        st.session_state.chat_manager.add_message(st.session_state.current_chat_id, assistant_response)

        # Display assistant message
        with st.chat_message("assistant"):
            st.write(assistant_response)
        
        # Update chat title with first user message if it's the default title
        chat_list = st.session_state.chat_manager.get_chat_list()
        current_chat = next((c for c in chat_list if c["id"] == st.session_state.current_chat_id), None)
        if current_chat and current_chat["title"].startswith("Chat "):
            # Use first few words of the first message as title
            title = " ".join(prompt.split()[:5])
            if len(prompt.split()) > 5:
                title += "..."
            st.session_state.chat_manager.update_chat_title(st.session_state.current_chat_id, title)
        
        st.rerun()

# Footer (only show if no active chat)
if st.session_state.current_chat_id is None:
    st.divider()
    st.caption("This is running locally and everything is AI generated.")