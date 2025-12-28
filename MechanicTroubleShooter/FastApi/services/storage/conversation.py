

import uuid
from datetime import datetime
from typing import Dict, List, Optional
from threading import Lock


class ConversationStore:
    
    def __init__(self, max_messages: int = 50):
        self._conversations: Dict[str, Dict] = {}
        self._lock = Lock()
        self._max_messages = max_messages
    
    def create_conversation(self) -> str:
        conv_id = str(uuid.uuid4())
        with self._lock:
            self._conversations[conv_id] = {
                "id": conv_id,
                "created_at": datetime.now().isoformat(),
                "messages": []
            }
        print(f"[CONV] Created conversation: {conv_id[:8]}...")
        return conv_id
    
    def add_message(self, conv_id: str, role: str, content: str) -> bool:
        with self._lock:
            if conv_id not in self._conversations:
                return False
            
            self._conversations[conv_id]["messages"].append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            })
            
            # Trim old messages if exceeding max
            if len(self._conversations[conv_id]["messages"]) > self._max_messages:
                self._conversations[conv_id]["messages"] = \
                    self._conversations[conv_id]["messages"][-self._max_messages:]
            
            return True
    
    def get_history(self, conv_id: str) -> Optional[List[Dict]]:
        with self._lock:
            if conv_id not in self._conversations:
                return None
            return self._conversations[conv_id]["messages"].copy()
    
    def get_or_create(self, conv_id: Optional[str]) -> str:
        if conv_id and conv_id in self._conversations:
            return conv_id
        return self.create_conversation()
    
    def clear_conversation(self, conv_id: str) -> bool:
        with self._lock:
            if conv_id in self._conversations:
                self._conversations[conv_id]["messages"] = []
                return True
            return False
    
    def delete_conversation(self, conv_id: str) -> bool:
        with self._lock:
            if conv_id in self._conversations:
                del self._conversations[conv_id]
                return True
            return False
    
    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "total_conversations": len(self._conversations),
                "total_messages": sum(
                    len(c["messages"]) for c in self._conversations.values()
                )
            }


# Singleton instance
_conversation_store: Optional[ConversationStore] = None


def get_conversation_store() -> ConversationStore:
    global _conversation_store
    if _conversation_store is None:
        _conversation_store = ConversationStore()
    return _conversation_store
