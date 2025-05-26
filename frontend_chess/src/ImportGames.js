import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Box, Typography, TextField, Button, List, ListItemButton, ListItemText, ThemeProvider, createTheme } from '@mui/material';

// Create dark theme
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
    text: {
      primary: '#ffffff',
      secondary: '#b3b3b3',
    },
  },
});

function ImportGames({ username }) {
  const [chessUsername, setChessUsername] = useState('');
  const [games, setGames] = useState([]);
  const [importMessage, setImportMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const handleImport = async (e) => {
    e.preventDefault();
    if (!chessUsername) {
      setImportMessage('Please enter your Chess.com username.');
      return;
    }
    setIsLoading(true);
    try {
      const response = await fetch(`http://localhost:5000/api/chesscom/games?username=${chessUsername}`);
      const data = await response.json();
      if (data.error) {
        setImportMessage(data.error);
      } else {
        const gameTitles = data.games.slice(-10).map((pgn, index) => ({
          title: `Game ${index + 1} - ${chessUsername}`,
          pgn,
        }));
        setGames(gameTitles);
        setImportMessage(`Imported ${gameTitles.length} game(s).`);
      }
    } catch (error) {
      console.error('Error importing games:', error);
      setImportMessage('Failed to import games.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleGameClick = async (game) => {
    const token = localStorage.getItem('authToken');
    if (!token) {
      setImportMessage('Please log in again.');
      navigate('/login');
      return;
    }

    const requestBody = {
      username,
      pgn: game.pgn,
      analysis: [],
      last_viewed_move: 0,
      comments: [],
    };

    try {
      const response = await fetch('http://localhost:5000/api/save-analysis', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify(requestBody),
      });
      
      const data = await response.json();
      if (!response.ok) {
        setImportMessage(data.error || 'Failed to save game');
        return;
      }
      setImportMessage('Game saved successfully!');
      navigate(`/analysis/${data.id}`);
    } catch (error) {
      console.error('Error saving analysis:', error);
      setImportMessage('Failed to contact server.');
    }
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <Box sx={{ 
        p: 4, 
        backgroundColor: 'background.default', 
        minHeight: '100vh',
        color: 'text.primary'
      }}>
        <Typography variant="h2" sx={{ mb: 2, color: 'text.primary' }}>
          Import Games from Chess.com
        </Typography>
        <Typography sx={{ color: 'text.secondary' }}>
          Logged in as: {username || 'Not logged in'}
        </Typography>
        <form onSubmit={handleImport}>
          <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
            <TextField
              type="text"
              placeholder="Enter Chess.com username"
              value={chessUsername}
              onChange={(e) => setChessUsername(e.target.value)}
              required
              fullWidth
              sx={{
                '& .MuiOutlinedInput-root': {
                  '& fieldset': {
                    borderColor: 'rgba(255, 255, 255, 0.23)',
                  },
                  '&:hover fieldset': {
                    borderColor: 'rgba(255, 255, 255, 0.5)',
                  },
                  '&.Mui-focused fieldset': {
                    borderColor: 'primary.main',
                  },
                },
                '& .MuiInputBase-input': {
                  color: 'text.primary',
                },
                '& .MuiInputLabel-root': {
                  color: 'text.secondary',
                },
                '& .MuiInputBase-input::placeholder': {
                  color: 'text.secondary',
                  opacity: 1,
                },
              }}
            />
            <Button 
              type="submit" 
              variant="contained" 
              disabled={isLoading}
              sx={{
                backgroundColor: 'primary.main',
                '&:hover': {
                  backgroundColor: 'primary.dark',
                },
                '&:disabled': {
                  backgroundColor: 'rgba(255, 255, 255, 0.12)',
                  color: 'rgba(255, 255, 255, 0.3)',
                },
              }}
            >
              {isLoading ? 'Importing...' : 'Import Games'}
            </Button>
          </Box>
        </form>
        <Typography sx={{ color: 'text.primary' }}>{importMessage}</Typography>
        {games.length > 0 && (
          <Box>
            <Typography variant="h3" sx={{ mt: 2, color: 'text.primary' }}>
              Imported Games:
            </Typography>
            <List sx={{ backgroundColor: 'background.paper', borderRadius: 1 }}>
              {games.map((game, index) => (
                <ListItemButton 
                  key={index} 
                  onClick={() => handleGameClick(game)}
                  sx={{
                    '&:hover': {
                      backgroundColor: 'rgba(255, 255, 255, 0.08)',
                    },
                  }}
                >
                  <ListItemText 
                    primary={game.title} 
                    sx={{ 
                      '& .MuiListItemText-primary': {
                        color: 'text.primary',
                      }
                    }}
                  />
                </ListItemButton>
              ))}
            </List>
          </Box>
        )}
      </Box>
    </ThemeProvider>
  );
}

export default ImportGames;