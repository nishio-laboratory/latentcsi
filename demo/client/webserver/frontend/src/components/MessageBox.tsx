import { useState } from "react";

function MessageBox() {
  const [msg, setMsg] = useState("");
  const handleMessageSend = () => {
    fetch("/control/msg", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value: msg }),
    });
  };

  return (
    <div className="flex items-center mt-5">
      <input
        type="text"
        placeholder="set_lr(0.001)"
        value={msg}
        onChange={(e) => setMsg(e.target.value)}
        className="bg-gray-300 rounded-l px-4 py-2 focus:outline-none"
      />
      <button
        onClick={() => handleMessageSend()}
        className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-r"
      >
        Send
      </button>
    </div>
  );
}

export default MessageBox;
