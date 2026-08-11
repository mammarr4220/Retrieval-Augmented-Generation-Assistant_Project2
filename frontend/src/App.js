import { useState } from "react";
import "./App.css";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  const askQuestion = async () => {
    if (!question.trim()) {
      return;
    }

    setLoading(true);
    setAnswer("");

    try {
      const response = await fetch("http://127.0.0.1:8000/api/ask/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: question,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get response from Django.");
      }

      const data = await response.json();

      setAnswer(data.answer);
    } catch (error) {
      setAnswer("Error: Could not connect to the Django backend.");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>HIV Service Delivery Assistant</h1>
        <p>Ask questions about the provided HIV guidelines.</p>
      </header>

      <main className="chat-container">
        <div className="question-section">
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask a question..."
            rows="4"
          />

          <button onClick={askQuestion} disabled={loading}>
            {loading ? "Asking..." : "Ask Question"}
          </button>
        </div>

        <div className="answer-section">
          <h2>Answer</h2>

          {loading ? (
            <p>Getting your answer...</p>
          ) : (
            <p>{answer || "Your answer will appear here."}</p>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;