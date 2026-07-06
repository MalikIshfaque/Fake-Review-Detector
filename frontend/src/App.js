import { useState, useEffect, useRef } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([
    {
      sender: "bot",
      text: "Heyy! 👋 I'm your AI Fake Review Detector! Paste any product review here, and I'll tell you if it's Real or Fake."
    }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [feedbackSent, setFeedbackSent] = useState({});  // track which msgIdx got feedback
  const chatContainerRef = useRef(null);

  const scrollToBottom = () => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTo({
        top: chatContainerRef.current.scrollHeight,
        behavior: "smooth"
      });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;
    
    const userMsg = { sender: "user", text: input };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const response = await axios.post("http://localhost:8000/predict", {
        review: userMsg.text,
      });
      
      const { result, confidence } = response.data;
      const isFake = result === "Fake";
      
      const botMsg = {
        sender: "bot",
        text: `This review is ${result}! (Confidence: ${confidence})`,
        isFake: isFake,
        showFeedback: true,
        reviewText: userMsg.text,
      };
      
      const followUpMsg = {
        sender: "bot",
        text: "Want me to check another one? Just paste it below! 👇"
      };

      setMessages(prev => [...prev, botMsg, followUpMsg]);
    } catch (err) {
      setMessages(prev => [...prev, {
        sender: "bot",
        text: "Oops! I couldn't connect to the server. Make sure FastAPI is running!"
      }]);
    }
    setLoading(false);
  };

  const handleFeedback = async (msgIdx, review, correctLabel) => {
    try {
      await axios.post("http://localhost:8000/feedback", {
        review: review,
        correct_label: correctLabel,
      });
      setFeedbackSent(prev => ({ ...prev, [msgIdx]: correctLabel }));
    } catch (err) {
      console.error("Feedback error:", err);
    }
  };

  return (
    <div className="app-container">
      {/* Background Lighting */}
      <div className="ambient-bg cyan"></div>
      <div className="ambient-bg red"></div>

      {/* Decorative Floating Glass Cards All Over The Page */}
      <div className="floating-card float-1">
        <div className="card-header">
          <div className="card-avatar">MF</div>
          <div>
            <h4 className="card-name">MAHA FALAK</h4>
            <p className="card-role">Verified Buyer</p>
          </div>
          <div className="card-stars">★★★★★</div>
        </div>
        <h3 className="card-title">Amazing Work!</h3>
        <p className="card-desc">I had a truly wonderful experience! I'm absolutely thrilled with the outstanding outcome of the project.</p>
      </div>

      <div className="floating-card float-2">
        <div className="card-header">
          <div className="card-avatar bg-purple">N</div>
          <div>
            <h4 className="card-name">NAJAF</h4>
            <p className="card-role">Customer</p>
          </div>
          <div className="card-stars">★★★★★</div>
        </div>
        <p className="card-desc" style={{marginTop: '12px'}}>The best service I've ever used. Completely flawless!</p>
      </div>

      <div className="floating-card float-3">
        <div className="card-header">
          <div className="card-avatar bg-orange">MI</div>
          <div>
            <h4 className="card-name">MALIK ISHFAQUE</h4>
            <p className="card-role">Business Owner</p>
          </div>
          <div className="card-stars">★★★★★</div>
        </div>
      </div>

      <div className="floating-card float-4">
         <div className="card-header">
          <div className="card-avatar bg-green">SA</div>
          <div>
            <h4 className="card-name">SARAH ALI</h4>
            <p className="card-role">Design Lead</p>
          </div>
          <div className="card-stars">★★★★★</div>
        </div>
        <p className="card-desc" style={{marginTop: '12px'}}>Super fast delivery and great support!</p>
      </div>

      <div className="floating-card float-5">
        <div className="card-header">
          <div className="card-avatar bg-pink">MK</div>
          <div>
            <h4 className="card-name">MICHAEL K.</h4>
            <p className="card-role">Tech Enthusiast</p>
          </div>
          <div className="card-stars">★★★★★</div>
        </div>
        <h3 className="card-title">Incredible!</h3>
      </div>

      <div className="floating-card float-10">
        <div className="card-header">
          <div className="card-avatar bg-cyan">HK</div>
          <div>
            <h4 className="card-name">HIMESH KUMAR</h4>
            <p className="card-role">Verified Buyer</p>
          </div>
          <div className="card-stars">★★★★★</div>
        </div>
        <p className="card-desc" style={{marginTop: '12px'}}>Exceptional quality! Exceeded all my expectations. Will definitely order again!</p>
      </div>

      <div className="floating-card float-6">
        <div className="card-header">
          <div className="card-avatar bg-yellow">AL</div>
          <div>
            <h4 className="card-name">ALEX LEO</h4>
            <p className="card-role">Verified Buyer</p>
          </div>
          <div className="card-stars">★★★★☆</div>
        </div>
        <p className="card-desc" style={{marginTop: '12px'}}>Really good, but the packaging could be slightly better.</p>
      </div>

      <div className="floating-card float-7">
        <div className="card-header">
          <div className="card-avatar bg-purple">EW</div>
          <div>
            <h4 className="card-name">EMMA WATSON</h4>
            <p className="card-role">Customer</p>
          </div>
          <div className="card-stars">★★★★★</div>
        </div>
        <p className="card-desc" style={{marginTop: '12px'}}>Absolutely fantastic! Highly recommended for everyone.</p>
      </div>

      <div className="floating-card float-8">
        <div className="card-header">
          <div className="card-avatar bg-cyan">DB</div>
          <div>
            <h4 className="card-name">DAVID BECK</h4>
            <p className="card-role">Business Owner</p>
          </div>
          <div className="card-stars">★★★★★</div>
        </div>
        <h3 className="card-title">Top Notch</h3>
      </div>

      <div className="floating-card float-9">
        <div className="card-header">
          <div className="card-avatar bg-red">LN</div>
          <div>
            <h4 className="card-name">LUCAS N.</h4>
            <p className="card-role">Verified Buyer</p>
          </div>
          <div className="card-stars">★★★★★</div>
        </div>
        <p className="card-desc" style={{marginTop: '12px'}}>Does exactly what it says on the tin. Perfect!</p>
      </div>

      {/* Main Chat Interface */}
      <div className="chat-window">
        <div className="chat-header">
          <div className="avatar">🤖</div>
          <div>
            <h2 className="header-title">Review Detector</h2>
            <p className="header-status">Online</p>
          </div>
        </div>
        
        <div className="chat-messages" ref={chatContainerRef}>
          {messages.map((msg, idx) => (
            <div key={idx} className={`message-wrapper ${msg.sender}`}>
              <div className={`message-bubble glass ${msg.sender} ${msg.isFake === true ? 'fake-result' : msg.isFake === false ? 'real-result' : ''}`}>
                {msg.text}
              </div>
              {msg.showFeedback && (
                <div className="feedback-row">
                  {feedbackSent[idx] ? (
                    <span className="feedback-thanks">
                      ✅ Thank you! Feedback saved.
                    </span>
                  ) : (
                    <>
                      <span className="feedback-label">Give feedback for better prediction next time. Was this correct?</span>
                      <button
                        className="feedback-btn correct"
                        onClick={() => handleFeedback(idx, msg.reviewText, msg.isFake ? "Fake" : "Real")}
                        title="Yes, prediction was correct"
                      >👍 Correct</button>
                      <button
                        className="feedback-btn wrong"
                        onClick={() => handleFeedback(idx, msg.reviewText, msg.isFake ? "Real" : "Fake")}
                        title="No, prediction was wrong"
                      >👎 Wrong</button>
                    </>
                  )}
                </div>
              )}
            </div>
          ))}
          {loading && (
            <div className="message-wrapper bot">
              <div className="message-bubble glass bot typing">
                <span>.</span><span>.</span><span>.</span>
              </div>
            </div>
          )}
          {/* Removed dummy div for old scrollIntoView */}
        </div>
        
        <div className="chat-input-area">
          <textarea
            className="chat-input glass"
            placeholder="Type or paste a review..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            rows={1}
          />
          <button className="send-btn glass" onClick={handleSend} disabled={loading || !input.trim()}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M2.01 21L23 12L2.01 3L2 10L17 12L2 14L2.01 21Z" fill="white"/>
            </svg>
          </button>
        </div>
      </div>

      {/* Developer Credit - Below Chat Window */}
      <div className="dev-credit" style={{ textAlign: 'center', lineHeight: '1.4' }}>
        <div>✦ Developed by <b>Malik Ishfaque</b></div>
        <div>© 2026 Malik Ishfaque. All rights reserved.</div>
      </div>
    </div>
  );
}

export default App;