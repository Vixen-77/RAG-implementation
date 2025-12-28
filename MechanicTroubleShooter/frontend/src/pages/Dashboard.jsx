import { useState, useEffect, useRef } from 'react';
import { useAuth } from '../context/AuthContext';
import { Send, Bot, User as UserIcon, Loader2, RefreshCw } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const Dashboard = () => {
  const { user } = useAuth();
  const [messages, setMessages] = useState([
    { id: 1, text: `Hello ${user?.username || 'there'}! I'm your AI Support Agent. How can I help you today?`, sender: 'bot' }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || isTyping) return;

    const userMessage = { id: Date.now(), text: input, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    const userQuery = input;
    setInput('');
    setIsTyping(true);

    // Create a placeholder for the bot response
    const botMessageId = Date.now() + 1;
    setMessages(prev => [...prev, { id: botMessageId, text: '', sender: 'bot', streaming: true }]);

    try {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      abortControllerRef.current = new AbortController();

      const response = await fetch('/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: userQuery,
          k: 8,
          conversation_id: conversationId
        }),
        signal: abortControllerRef.current.signal
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let fullResponse = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('event:')) {
            const eventType = line.slice(6).trim();
            continue;
          }
          if (line.startsWith('data:')) {
            const data = line.slice(5);  // Don't trim - spaces are valid tokens!

            // Try to parse as JSON (metadata event)
            try {
              const parsed = JSON.parse(data);
              if (parsed.conversation_id) {
                setConversationId(parsed.conversation_id);
              }
            } catch {
              // It's a token, append to response (skip empty strings only)
              if (data !== '') {
                fullResponse += data;
                setMessages(prev => prev.map(msg =>
                  msg.id === botMessageId
                    ? { ...msg, text: fullResponse }
                    : msg
                ));
              }
            }
          }
        }
      }

      // Mark streaming complete
      setMessages(prev => prev.map(msg =>
        msg.id === botMessageId
          ? { ...msg, streaming: false }
          : msg
      ));

    } catch (error) {
      if (error.name === 'AbortError') return;
      console.error('Chat error:', error);
      setMessages(prev => prev.map(msg =>
        msg.id === botMessageId
          ? { ...msg, text: 'Sorry, an error occurred. Please try again.', streaming: false }
          : msg
      ));
    } finally {
      setIsTyping(false);
    }
  };

  const handleNewConversation = () => {
    setConversationId(null);
    setMessages([
      { id: Date.now(), text: `Hello ${user?.username || 'there'}! Starting a new conversation. How can I help you?`, sender: 'bot' }
    ]);
  };

  return (
    <div className="min-h-screen pt-20 px-4 pb-4 flex flex-col items-center">
      <div className="max-w-4xl w-full flex-1 flex flex-col glass-card h-[85vh] overflow-hidden">
        {/* Header */}
        <div className="p-4 border-b border-white/10 flex items-center justify-between bg-white/5">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-primary to-accent flex items-center justify-center">
              <Bot className="text-white w-6 h-6" />
            </div>
            <div>
              <h3 className="font-semibold text-white">AI Support Agent</h3>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                <span className="text-xs text-gray-400">Online</span>
              </div>
            </div>
          </div>
          <button
            onClick={handleNewConversation}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors text-gray-400 hover:text-white"
            title="New conversation"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
        </div>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
          {messages.map((msg) => (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              key={msg.id}
              className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`flex items-end gap-2 max-w-[80%] ${msg.sender === 'user' ? 'flex-row-reverse' : ''}`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${msg.sender === 'user' ? 'bg-indigo-600' : 'bg-gray-700'}`}>
                  {msg.sender === 'user' ? <UserIcon className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                </div>
                <div className={`p-3 rounded-2xl ${msg.sender === 'user' ? 'bg-indigo-600 text-white rounded-br-none' : 'bg-white/10 text-gray-200 rounded-bl-none'}`}>
                  {msg.text}
                  {msg.streaming && <span className="inline-block w-1 h-4 bg-gray-400 ml-1 animate-pulse" />}
                </div>
              </div>
            </motion.div>
          ))}
          {isTyping && messages[messages.length - 1]?.sender !== 'bot' && (
            <div className="flex justify-start">
              <div className="flex items-end gap-2">
                <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center flex-shrink-0">
                  <Bot className="w-4 h-4" />
                </div>
                <div className="bg-white/10 p-3 rounded-2xl rounded-bl-none">
                  <Loader2 className="w-4 h-4 animate-spin text-gray-400" />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-4 border-t border-white/10 bg-white/5">
          <form onSubmit={handleSend} className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              className="glass-input bg-transparent"
              disabled={isTyping}
            />
            <button
              type="submit"
              className="p-3 bg-primary hover:bg-indigo-600 rounded-lg transition-colors text-white disabled:opacity-50"
              disabled={!input.trim() || isTyping}
            >
              <Send className="w-5 h-5" />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;

