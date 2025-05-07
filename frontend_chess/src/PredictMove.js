import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  Button, 
  Input,
  Paper,
  Container,
  createTheme,
  ThemeProvider,
  CircularProgress,
  Divider
} from '@mui/material';
import { Chessboard } from 'react-chessboard';
import { styled } from '@mui/material/styles';

// Create a dark chess theme
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9', // Light blue
    },
    secondary: {
      main: '#ce93d8', // Light purple
    },
    background: {
      default: '#121212', // Very dark gray
      paper: '#1e1e1e',   // Dark gray
    },
    text: {
      primary: '#e0e0e0',
      secondary: '#b0b0b0',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h2: {
      fontWeight: 500,
    },
    h4: {
      fontWeight: 500,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
          padding: '10px 24px',
        },
        containedPrimary: {
          background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
          boxShadow: '0 3px 5px 2px rgba(33, 203, 243, .3)',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          padding: 24,
        },
      },
    },
  },
});

// Styled file input component
const VisuallyHiddenInput = styled('input')({
  clip: 'rect(0 0 0 0)',
  clipPath: 'inset(50%)',
  height: 1,
  overflow: 'hidden',
  position: 'absolute',
  bottom: 0,
  left: 0,
  whiteSpace: 'nowrap',
  width: 1,
});

// Styled components for the chess board
const ChessResultContainer = styled(Box)(({ theme }) => ({
  marginTop: theme.spacing(3),
  padding: theme.spacing(2),
  borderRadius: 8,
  background: theme.palette.background.paper,
  boxShadow: '0 8px 16px rgba(0, 0, 0, 0.3)',
}));

const ResultItem = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  padding: theme.spacing(1.5),
  borderRadius: 4,
  backgroundColor: 'rgba(255, 255, 255, 0.05)',
  marginBottom: theme.spacing(1),
}));

function PredictMove() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState('');

  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      setFile(e.target.files[0]);
      setFileName(e.target.files[0].name);
    }
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!file) return;
    
    setIsLoading(true);
    const formData = new FormData();
    formData.append('image', file);
    
    try {
      const response = await fetch('http://localhost:5000/api/predict_move', {
        method: 'POST',
        body: formData,
      });
      const resData = await response.json();
      setResult(resData);
    } catch (error) {
      console.error('Prediction error:', error);
      setResult({ error: 'Prediction failed' });
    } finally {
      setIsLoading(false);
    }
  };

  // Custom chess board theme
  const customBoardTheme = {
    boardStyle: {
      borderRadius: '4px',
      boxShadow: '0 5px 15px rgba(0, 0, 0, 0.5)',
    },
    darkSquareStyle: { backgroundColor: '#769656' },
    lightSquareStyle: { backgroundColor: '#eeeed2' },
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <Box sx={{ 
        minHeight: '100vh', 
        bgcolor: 'background.default', 
        py: 5,
        backgroundImage: 'radial-gradient(circle at 50% 50%, #1e1e2e 0%, #121212 100%)',
      }}>
        <Container maxWidth="md">
          <Paper elevation={6} sx={{ py: 4, px: 3 }}>
            <Typography variant="h2" sx={{ mb: 4, textAlign: 'center', fontWeight: 'bold', color: 'primary.main' }}>
              Chess Move Predictor
            </Typography>
            
            <Box component="form" onSubmit={handlePredict} sx={{ textAlign: 'center' }}>
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mb: 3 }}>
                <Button
                  component="label"
                  variant="outlined"
                  sx={{ mb: 2, px: 4, py: 1.5 }}
                  startIcon={<span>📷</span>}
                >
                  Select Chessboard Image
                  <VisuallyHiddenInput 
                    type="file" 
                    onChange={handleFileChange} 
                    accept="image/*" 
                  />
                </Button>
                
                {fileName && (
                  <Typography variant="body2" color="text.secondary">
                    Selected: {fileName}
                  </Typography>
                )}
              </Box>
              
              <Button 
                type="submit" 
                variant="contained" 
                disabled={isLoading || !file}
                size="large"
                sx={{ minWidth: 200 }}
              >
                {isLoading ? (
                  <CircularProgress size={24} color="inherit" />
                ) : (
                  'Analyze Position'
                )}
              </Button>
            </Box>

            {result && (
              <ChessResultContainer>
                <Typography variant="h4" sx={{ mb: 3, color: 'primary.light' }}>
                  Analysis Result
                </Typography>
                
                <Divider sx={{ mb: 3 }} />
                
                {result.error ? (
                  <Typography color="error" sx={{ textAlign: 'center', py: 3 }}>
                    {result.error}
                  </Typography>
                ) : (
                  <>
                    <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 4 }}>
                      <Box sx={{ flex: 1, minWidth: { xs: '100%', md: '350px' } }}>
                        <Chessboard 
                          position={result.fen} 
                          boardWidth={350}
                          customBoardStyle={customBoardTheme.boardStyle}
                          customDarkSquareStyle={customBoardTheme.darkSquareStyle}
                          customLightSquareStyle={customBoardTheme.lightSquareStyle}
                        />
                      </Box>
                      <Box sx={{ flex: 1 }}>
                        <ResultItem>
                          <Typography variant="subtitle1" sx={{ width: '120px', fontWeight: 'bold' }}>
                            Best Move:
                          </Typography>
                          <Typography 
                            variant="h5" 
                            sx={{ 
                              fontWeight: 'bold', 
                              color: 'secondary.light',
                              ml: 2
                            }}
                          >
                            {result.best_move}
                          </Typography>
                        </ResultItem>
                        
                        <ResultItem>
                          <Typography variant="subtitle1" sx={{ width: '120px', fontWeight: 'bold' }}>
                            Evaluation:
                          </Typography>
                          <Typography 
                            variant="h6" 
                            sx={{ 
                              fontWeight: 'bold', 
                              color: Number(result.evaluation) > 0 ? '#4caf50' : '#f44336',
                              ml: 2
                            }}
                          >
                            {result.evaluation.toFixed(2)}
                          </Typography>
                        </ResultItem>
                        
                        <ResultItem>
                          <Typography variant="subtitle1" sx={{ width: '120px', fontWeight: 'bold' }}>
                            FEN:
                          </Typography>
                          <Typography 
                            variant="body2" 
                            sx={{ 
                              fontFamily: 'monospace', 
                              bgcolor: 'rgba(0,0,0,0.2)',
                              p: 1,
                              borderRadius: 1,
                              overflow: 'auto',
                              ml: 2
                            }}
                          >
                            {result.fen}
                          </Typography>
                        </ResultItem>
                      </Box>
                    </Box>
                  </>
                )}
              </ChessResultContainer>
            )}
          </Paper>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default PredictMove;