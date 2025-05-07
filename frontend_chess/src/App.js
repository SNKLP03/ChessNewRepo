import React, { useState } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import Register from "./components/Register";
import Login from "./components/Login";
import Dashboard from "./components/Dashboard";
import GameAnalysis from "./GameAnalysis";
import ImportGames from "./ImportGames";
import PredictMove from "./PredictMove";

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState("");

  const handleLoginSuccess = (user) => {
    setIsLoggedIn(true);
    setUsername(user);
  };

  return (
    <Router>
      <Routes>
        <Route path="/register" element={<Register />} />
        <Route
          path="/login"
          element={<Login onLoginSuccess={handleLoginSuccess} />}
        />
        <Route path="/dashboard" element={<Dashboard username={username} />} />
        <Route path="/predict" element={<PredictMove />} />
        <Route path="/import" element={<ImportGames username={username} />} />
        {/* GameAnalysis routes - No auth check */}
        <Route
          path="/analysis/:analysisId"
          element={<GameAnalysis username={username} />}
        />
        <Route
          path="/analysis/new"
          element={<GameAnalysis username={username} />}
        />
        {/* Default redirect to login */}
        <Route path="/" element={<Navigate to="/login" />} />
      </Routes>
    </Router>
  );
}

export default App;