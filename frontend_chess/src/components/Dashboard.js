import React, { useState, useEffect } from "react";
import { Link, Outlet, useNavigate } from "react-router-dom";
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  Button,
  Grid,
  Card,
  CardContent,
  CardActions,
  TextField,
  List,
  Menu,
  MenuItem,
  ListItem,
  ListItemButton,
  ListItemText,
  CssBaseline,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Avatar,
  LinearProgress,
} from "@mui/material";
import { AccountCircle } from "@mui/icons-material";
import { Info, ExitToApp } from "@mui/icons-material";
import { styled } from "@mui/material/styles";
import { createTheme, ThemeProvider } from "@mui/material/styles";
import {
  Logout as LogoutIcon,
  History as HistoryIcon,
  CloudUpload as CloudUploadIcon,
  SportsEsports as SportsEsportsIcon,
  MenuBook as MenuBookIcon,
  Search as SearchIcon,
  Psychology as PsychologyIcon,
  Casino as ChessIcon,
  Info as InfoIcon,
} from "@mui/icons-material";

// Dark game theme with vibrant accents
const darkTheme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#9c27b0", // Purple
    },
    secondary: {
      main: "#2196f3", // Blue
    },
    error: {
      main: "#f44336", // Red
    },
    warning: {
      main: "#ff9800", // Orange
    },
    info: {
      main: "#00bcd4", // Cyan
    },
    success: {
      main: "#4caf50", // Green
    },
    background: {
      default: "#121212",
      paper: "#1e1e1e",
    },
    text: {
      primary: "#ffffff",
      secondary: "#b3b3b3",
    },
  },
  typography: {
    fontFamily: '"Orbitron", "Roboto", sans-serif',
    h4: {
      fontWeight: 700,
      letterSpacing: "1px",
    },
    h5: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: "none",
          borderRadius: 8,
          padding: "8px 16px",
          fontWeight: 600,
          transition: "all 0.3s ease",
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          transition: "all 0.3s ease",
          "&:hover": {
            transform: "translateY(-5px)",
          },
        },
      },
    },
  },
});

// Glowing text component
const GlowingText = styled(Typography)(({ theme, glowcolor = "#9c27b0" }) => ({
  color: "#ffffff",
  textShadow: `0 0 10px ${glowcolor}, 0 0 20px rgba(156, 39, 176, 0.5)`,
  fontWeight: 700,
}));

// Gradient button component
const GradientButton = styled(Button)(({ theme }) => ({
  background: "linear-gradient(45deg, #9c27b0 30%, #2196f3 90%)",
  border: 0,
  color: "white",
  padding: "8px 16px",
  boxShadow: "0 3px 5px rgba(156, 39, 176, 0.3)",
  "&:hover": {
    boxShadow: "0 6px 10px rgba(156, 39, 176, 0.4)",
    transform: "translateY(-2px)",
  },
}));

// Styled card with animated border
const GameCard = styled(Card)(({ theme }) => ({
  backgroundColor: "#1e1e1e",
  borderRadius: 16,
  boxShadow: "0 8px 32px rgba(0, 0, 0, 0.3)",
  border: "1px solid rgba(255, 255, 255, 0.05)",
  position: "relative",
  overflow: "hidden",
  "&:before": {
    content: '""',
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    height: "4px",
    background: "linear-gradient(90deg, #9c27b0, #2196f3)",
    animation: "borderAnimation 3s linear infinite",
    backgroundSize: "200% 200%",
  },
  "@keyframes borderAnimation": {
    "0%": { backgroundPosition: "0% 50%" },
    "50%": { backgroundPosition: "100% 50%" },
    "100%": { backgroundPosition: "0% 50%" },
  },
}));

function Dashboard({ username }) {
  const navigate = useNavigate();
  const [importName, setImportName] = useState("");
  const [importedGames, setImportedGames] = useState([]);
  const [importError, setImportError] = useState("");
  const [analysisHistory, setAnalysisHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [openPopup, setOpenPopup] = useState(false);

  useEffect(() => {
    if (username) {
      fetchAnalysisHistory();
    } else {
      console.error("Username is undefined, skipping fetchAnalysisHistory");
      setImportError("Please log in to view analysis history.");
    }
  }, [username]);

  const fetchAnalysisHistory = async () => {
    try {
      const token = localStorage.getItem('authToken');
      if (!token) {
        throw new Error("No authentication token found. Please log in.");
      }
      const response = await fetch(
        `http://localhost:5000/api/analysis-history/${username}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
        }
      );
      const data = await response.json();
      if (response.ok) {
        setAnalysisHistory(data.history || []);
        if (data.history.length === 0) {
          setOpenPopup(true);
        }
      } else {
        console.error('Error fetching analysis history:', data.error);
        setImportError(data.error || "Failed to fetch analysis history.");
      }
    } catch (error) {
      console.error("Error fetching analysis history:", error.message);
      setImportError("Failed to fetch analysis history. Please try again.");
    }
  };

  const handleGameAnalysisClick = () => {
    if (analysisHistory.length > 0) {
      const latestAnalysis = analysisHistory[0];
      navigate(`/analysis/${latestAnalysis.id}`);
    } else {
      setOpenPopup(true);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("authToken");
    navigate("/login");
  };

  const handleImportGame = async () => {
    if (!importName.trim()) {
      setImportError("Please enter a username.");
      return;
    }
    setIsLoading(true);
    try {
      const response = await fetch(
        `http://localhost:5000/api/chesscom/games?username=${importName.trim()}`,
        {
          method: "GET",
          headers: { "Content-Type": "application/json" },
        }
      );
      const data = await response.json();
      if (response.ok && data.games && data.games.length > 0) {
        const gameTitles = data.games.slice(-10).map((pgn, index) => ({
          title: `Game ${index + 1} - ${importName}`,
          pgn,
        }));
        setImportedGames(gameTitles);
        setImportName("");
        setImportError("");
      } else {
        setImportError(data.error || "No games found for this username.");
      }
    } catch (error) {
      console.error("Error importing games:", error);
      setImportError("Failed to fetch games from Chess.com.");
    } finally {
      setIsLoading(false);
    }
  };

  const [anchorEl, setAnchorEl] = useState(null);
  const open = Boolean(anchorEl);

  const handleMenu = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box
        sx={{
          minHeight: "100vh",
          background:
            "radial-gradient(circle at center, #1a1a2e 0%, #121212 100%)",
          position: "relative",
          overflow: "hidden",
          "&:before": {
            content: '""',
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundImage:
              "url(\"data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%239C27B0' fill-opacity='0.05' fill-rule='evenodd'/%3E%3C/svg%3E\")",
            opacity: 0.3,
            zIndex: 0,
          },
        }}
      >
        {/* Floating chess pieces animation */}
        <Box
          sx={{
            position: "absolute",
            top: "10%",
            left: "5%",
            animation: "float 6s ease-in-out infinite",
            opacity: 0.1,
            zIndex: 0,
            "@keyframes float": {
              "0%, 100%": { transform: "translateY(0)" },
              "50%": { transform: "translateY(-20px)" },
            },
          }}
        >
          <ChessIcon sx={{ fontSize: 120 }} />
        </Box>

        <AppBar
          position="static"
          sx={{
            background: "rgba(30, 30, 30, 0.8)",
            backdropFilter: "blur(10px)",
            boxShadow: "0 4px 30px rgba(0, 0, 0, 0.3)",
            borderBottom: "1px solid rgba(156, 39, 176, 0.2)",
          }}
        >
          <Toolbar sx={{ justifyContent: "space-between" }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
              <ChessIcon
                sx={{
                  fontSize: 36,
                  color: "#9c27b0",
                  filter: "drop-shadow(0 0 8px rgba(156, 39, 176, 0.7))",
                }}
              />
              <GlowingText variant="h5" glowcolor="#2196f3">
                CHESS MASTER
              </GlowingText>
            </Box>
            <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
              <IconButton
                onClick={handleMenu}
                size="small"
                sx={{ ml: 2 }}
                aria-controls={open ? "account-menu" : undefined}
                aria-haspopup="true"
                aria-expanded={open ? "true" : undefined}
              >
                <Avatar
                  sx={{
                    background:
                      "linear-gradient(45deg, #9c27b0 30%, #2196f3 90%)",
                    boxShadow: "0 0 10px rgba(156, 39, 176, 0.5)",
                  }}
                >
                  {username ? username.charAt(0).toUpperCase() : "?"}
                </Avatar>
              </IconButton>

              <Menu
                anchorEl={anchorEl}
                id="account-menu"
                open={open}
                onClose={handleClose}
                onClick={handleClose}
                PaperProps={{
                  elevation: 0,
                  sx: {
                    overflow: "visible",
                    filter: "drop-shadow(0px 2px 8px rgba(0,0,0,0.32))",
                    mt: 1.5,
                    background:
                      "linear-gradient(135deg, rgba(30, 30, 30, 0.95) 0%, rgba(18, 18, 18, 0.95) 100%)",
                    border: "1px solid rgba(156, 39, 176, 0.3)",
                    "& .MuiAvatar-root": {
                      width: 32,
                      height: 32,
                      ml: -0.5,
                      mr: 1,
                    },
                    "&:before": {
                      content: '""',
                      display: "block",
                      position: "absolute",
                      top: 0,
                      right: 14,
                      width: 10,
                      height: 10,
                      bgcolor: "background.paper",
                      transform: "translateY(-50%) rotate(45deg)",
                      zIndex: 0,
                      background: "rgba(30, 30, 30, 0.95)",
                      borderLeft: "1px solid rgba(156, 39, 176, 0.3)",
                      borderTop: "1px solid rgba(156, 39, 176, 0.3)",
                    },
                  },
                }}
                transformOrigin={{ horizontal: "right", vertical: "top" }}
                anchorOrigin={{ horizontal: "right", vertical: "bottom" }}
              >
                <MenuItem onClick={handleClose}>
                  <AccountCircle sx={{ mr: 1, color: "#9c27b0" }} />
                  <Typography>Profile</Typography>
                </MenuItem>
                <MenuItem onClick={handleClose}>
                  <Info sx={{ mr: 1, color: "#2196f3" }} />
                  <Typography>About</Typography>
                </MenuItem>
                <MenuItem
                  onClick={() => {
                    handleClose();
                    handleLogout();
                  }}
                >
                  <ExitToApp sx={{ mr: 1, color: "#f44336" }} />
                  <Typography>Logout</Typography>
                </MenuItem>
              </Menu>
            </Box>
          </Toolbar>
        </AppBar>

        <Container
          maxWidth="lg"
          sx={{ mt: 4, mb: 4, position: "relative", zIndex: 1 }}
        >
          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              mb: 6,
              textAlign: "center",
            }}
          >
            <GlowingText variant="h4" glowcolor="#9c27b0">
              Welcome to Your Chess Arena, {username || "Guest"}!
            </GlowingText>
            <Typography
              variant="subtitle1"
              sx={{
                color: "#b3b3b3",
                mt: 1,
                textShadow: "0 0 5px rgba(255, 255, 255, 0.2)",
              }}
            >
              Analyze, Predict, and Master Your Chess Game
            </Typography>
          </Box>

          <Grid container spacing={3}>
            {/* Game Analysis Card */}
            <Grid item xs={12} md={8}>
              <GameCard sx={{ height: 400 }}>
                <Box
                  sx={{
                    height: "60%",
                    background:
                      "linear-gradient(135deg, rgba(33, 150, 243, 0.1) 0%, rgba(156, 39, 176, 0.1) 100%)",
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                    borderBottom: "1px solid rgba(255, 255, 255, 0.05)",
                    position: "relative",
                    overflow: "hidden",
                    "&:after": {
                      content: '""',
                      position: "absolute",
                      top: 0,
                      left: 0,
                      right: 0,
                      bottom: 0,
                      backgroundImage:
                        'url("https://images.unsplash.com/photo-1560174038-da43ac74f01b?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80")',
                      backgroundSize: "cover",
                      backgroundPosition: "center",
                      opacity: 0.15,
                    },
                  }}
                >
                  <SportsEsportsIcon
                    sx={{
                      fontSize: 80,
                      color: "rgba(255, 255, 255, 0.2)",
                      filter: "drop-shadow(0 0 10px rgba(156, 39, 176, 0.5))",
                    }}
                  />
                </Box>
                <CardContent>
                  <Typography
                    variant="h5"
                    sx={{ color: "#fff", fontWeight: 600 }}
                  >
                    Game Analysis
                  </Typography>
                  <Typography variant="body2" sx={{ color: "#b3b3b3", mt: 1 }}>
                    Deep dive into your games with powerful AI analysis to
                    identify patterns and improve your strategy.
                  </Typography>
                </CardContent>
                <CardActions
                  sx={{ justifyContent: "center", mt: "auto", mb: 2 }}
                >
                  <GradientButton
                    onClick={handleGameAnalysisClick}
                    startIcon={<SportsEsportsIcon />}
                  >
                    Analyze Now
                  </GradientButton>
                </CardActions>
              </GameCard>
            </Grid>

            {/* Analysis History Card */}
            <Grid item xs={12} md={4}>
              <GameCard sx={{ height: 400 }}>
                <CardContent>
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      gap: 1,
                      mb: 2,
                    }}
                  >
                    <HistoryIcon sx={{ color: "#2196f3" }} />
                    <Typography
                      variant="h6"
                      sx={{ color: "#fff", fontWeight: 600 }}
                    >
                      Analysis History
                    </Typography>
                  </Box>
                  <Box
                    sx={{
                      height: 300,
                      overflowY: "auto",
                      "&::-webkit-scrollbar": {
                        width: "6px",
                      },
                      "&::-webkit-scrollbar-track": {
                        background: "rgba(255, 255, 255, 0.05)",
                        borderRadius: "10px",
                      },
                      "&::-webkit-scrollbar-thumb": {
                        background: "rgba(156, 39, 176, 0.5)",
                        borderRadius: "10px",
                      },
                    }}
                  >
                    <List>
                      {analysisHistory.length > 0 ? (
                        analysisHistory.map((entry, index) => (
                          <ListItem key={entry.id} disablePadding>
                            <ListItemButton
                              onClick={() => navigate(`/analysis/${entry.id}`)}
                              sx={{
                                borderRadius: "8px",
                                mb: 1,
                                transition: "all 0.2s ease",
                                "&:hover": {
                                  background: "rgba(156, 39, 176, 0.1)",
                                  transform: "translateX(4px)",
                                },
                              }}
                            >
                              <ListItemText
                                primary={`Game ${index + 1} - ${new Date(
                                  entry.timestamp
                                ).toLocaleDateString()}`}
                                secondary={`Last viewed: Move ${
                                  entry.last_viewed_move || 0
                                }`}
                                primaryTypographyProps={{
                                  fontWeight: 500,
                                  color: "#fff",
                                }}
                                secondaryTypographyProps={{
                                  color: "#b3b3b3",
                                  fontSize: "0.8rem",
                                }}
                              />
                            </ListItemButton>
                          </ListItem>
                        ))
                      ) : (
                        <Box
                          sx={{
                            height: "100%",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            flexDirection: "column",
                            opacity: 0.5,
                          }}
                        >
                          <HistoryIcon
                            sx={{ fontSize: 48, color: "#b3b3b3", mb: 2 }}
                          />
                          <Typography
                            variant="body2"
                            sx={{ color: "#b3b3b3", textAlign: "center" }}
                          >
                            No analysis history found
                          </Typography>
                        </Box>
                      )}
                    </List>
                  </Box>
                </CardContent>
              </GameCard>
            </Grid>

            {/* Import Games Card */}
            <Grid item xs={12} md={8}>
              <GameCard sx={{ height: 300 }}>
                <CardContent>
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      gap: 1,
                      mb: 2,
                    }}
                  >
                    <CloudUploadIcon sx={{ color: "#9c27b0" }} />
                    <Typography
                      variant="h6"
                      sx={{ color: "#fff", fontWeight: 600 }}
                    >
                      Import Games from Chess.com
                    </Typography>
                  </Box>
                  <Box sx={{ display: "flex", gap: 1, mt: 1 }}>
                    <TextField
                      label="Chess.com Username"
                      variant="outlined"
                      size="small"
                      value={importName}
                      onChange={(e) => setImportName(e.target.value)}
                      sx={{
                        flexGrow: 1,
                        "& .MuiOutlinedInput-root": {
                          "& fieldset": {
                            borderColor: "rgba(156, 39, 176, 0.3)",
                          },
                          "&:hover fieldset": {
                            borderColor: "rgba(156, 39, 176, 0.5)",
                          },
                          "&.Mui-focused fieldset": {
                            borderColor: "#9c27b0",
                          },
                        },
                      }}
                    />
                    <GradientButton
                      onClick={handleImportGame}
                      disabled={isLoading}
                      startIcon={<CloudUploadIcon />}
                    >
                      {isLoading ? "Importing..." : "Import"}
                    </GradientButton>
                  </Box>
                  {isLoading && (
                    <Box sx={{ width: "100%", mt: 2 }}>
                      <LinearProgress
                        sx={{
                          "& .MuiLinearProgress-bar": {
                            background:
                              "linear-gradient(45deg, #9c27b0 30%, #2196f3 90%)",
                          },
                          height: 6,
                          borderRadius: 3,
                        }}
                      />
                    </Box>
                  )}
                  {importError && (
                    <Typography
                      variant="body2"
                      sx={{ color: "#f44336", mt: 1 }}
                    >
                      {importError}
                    </Typography>
                  )}
                  <Box
                    sx={{
                      maxHeight: 150,
                      overflowY: "auto",
                      mt: 2,
                    }}
                  >
                    <List>
                      {importedGames.map((game, index) => (
                        <ListItem key={index} disablePadding>
                          <ListItemButton
                            onClick={async () => {
                              const token = localStorage.getItem('authToken');
                              const response = await fetch(
                                "http://localhost:5000/api/save-analysis",
                                {
                                  method: "POST",
                                  headers: {
                                    "Content-Type": "application/json",
                                    "Authorization": `Bearer ${token}`,
                                  },
                                  body: JSON.stringify({
                                    username,
                                    pgn: game.pgn,
                                    analysis: [],
                                    last_viewed_move: 0,
                                    comments: [],
                                  }),
                                }
                              );
                              const data = await response.json();
                              if (response.ok) {
                                navigate(`/analysis/${data.id}`);
                              } else {
                                setImportError(data.error || 'Failed to save analysis');
                              }
                            }}
                            sx={{
                              borderRadius: "8px",
                              mb: 1,
                              background: "rgba(33, 150, 243, 0.05)",
                              transition: "all 0.2s ease",
                              "&:hover": {
                                background: "rgba(33, 150, 243, 0.1)",
                              },
                            }}
                          >
                            <ListItemText
                              primary={game.title}
                              primaryTypographyProps={{
                                fontWeight: 500,
                                color: "#fff",
                              }}
                            />
                          </ListItemButton>
                        </ListItem>
                      ))}
                    </List>
                  </Box>
                </CardContent>
              </GameCard>
            </Grid>

            {/* Predict Move Card */}
            <Grid item xs={12} md={4}>
              <GameCard sx={{ height: 300 }}>
                <CardContent>
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      gap: 1,
                      mb: 2,
                    }}
                  >
                    <PsychologyIcon sx={{ color: "#00bcd4" }} />
                    <Typography
                      variant="h6"
                      sx={{ color: "#fff", fontWeight: 600 }}
                    >
                      Predict Move
                    </Typography>
                  </Box>
                  <Typography variant="body2" sx={{ color: "#b3b3b3", mt: 1 }}>
                    Upload an image of a chessboard to predict the best possible
                    move using our advanced AI engine.
                  </Typography>
                  <Box
                    sx={{
                      display: "flex",
                      justifyContent: "center",
                      alignItems: "center",
                      mt: 3,
                      height: "100px",
                      background: "rgba(255, 255, 255, 0.03)",
                      borderRadius: "8px",
                      border: "1px dashed rgba(255, 255, 255, 0.1)",
                    }}
                  >
                    <CloudUploadIcon
                      sx={{
                        fontSize: 40,
                        color: "rgba(255, 255, 255, 0.2)",
                        animation: "pulse 2s infinite",
                        "@keyframes pulse": {
                          "0%": { opacity: 0.2 },
                          "50%": { opacity: 0.6 },
                          "100%": { opacity: 0.2 },
                        },
                      }}
                    />
                  </Box>
                </CardContent>
                <CardActions sx={{ justifyContent: "center", mt: "auto" }}>
                  <Link to="/predict" style={{ textDecoration: "none" }}>
                    <GradientButton startIcon={<PsychologyIcon />}>
                      Predict Now
                    </GradientButton>
                  </Link>
                </CardActions>
              </GameCard>
            </Grid>

            {/* Learn from Grand Masters Card */}
            <Grid item xs={12} md={6}>
              <GameCard sx={{ height: 220 }}>
                <CardContent>
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      gap: 1,
                      mb: 2,
                    }}
                  >
                    <SportsEsportsIcon sx={{ color: "#ff9800" }} />
                    <Typography
                      variant="h6"
                      sx={{ color: "#fff", fontWeight: 600 }}
                    >
                      Learn from Grand Masters
                    </Typography>
                  </Box>
                  <Typography variant="body2" sx={{ color: "#b3b3b3", mt: 1 }}>
                    Study advanced moves, strategies and tactics from the
                    world's greatest chess players.
                  </Typography>
                </CardContent>
                <CardActions sx={{ justifyContent: "center", mt: "auto" }}>
                  <Button
                    component={Link}
                    to="learn-grandmaster"
                    variant="outlined"
                    sx={{
                      color: "#ff9800",
                      borderColor: "#ff9800",
                      "&:hover": {
                        borderColor: "#ffb74d",
                        background: "rgba(255, 152, 0, 0.1)",
                      },
                    }}
                    startIcon={<SportsEsportsIcon />}
                  >
                    Start Learning
                  </Button>
                </CardActions>
              </GameCard>
            </Grid>

            {/* Find & Learn Card */}
            <Grid item xs={12} md={3}>
              <GameCard sx={{ height: 220 }}>
                <CardContent>
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      gap: 1,
                      mb: 2,
                    }}
                  >
                    <SearchIcon sx={{ color: "#4caf50" }} />
                    <Typography
                      variant="h6"
                      sx={{ color: "#fff", fontWeight: 600 }}
                    >
                      Play & Learn
                    </Typography>
                  </Box>
                  <Typography variant="body2" sx={{ color: "#b3b3b3", mt: 1 }}>
                    Discover new games, openings, and interactive chess lessons
                    tailored to your skill level.
                  </Typography>
                </CardContent>
                <CardActions sx={{ justifyContent: "center", mt: "auto" }}>
                  <Button
                    component={Link}
                    to="find-learn"
                    variant="outlined"
                    sx={{
                      color: "#4caf50",
                      borderColor: "#4caf50",
                      "&:hover": {
                        borderColor: "#81c784",
                        background: "rgba(76, 175, 80, 0.1)",
                      },
                    }}
                    startIcon={<SearchIcon />}
                  >
                    Play Now with AI
                  </Button>
                </CardActions>
              </GameCard>
            </Grid>

            {/* Chess Books Card */}
            <Grid item xs={12} md={3}>
              <GameCard sx={{ height: 220 }}>
                <CardContent>
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      gap: 1,
                      mb: 2,
                    }}
                  >
                    <MenuBookIcon sx={{ color: "#9c27b0" }} />
                    <Typography
                      variant="h6"
                      sx={{ color: "#fff", fontWeight: 600 }}
                    >
                      Chess Books
                    </Typography>
                  </Box>
                  <Typography variant="body2" sx={{ color: "#b3b3b3", mt: 1 }}>
                    Explore recommended chess literature to deepen your
                    understanding of the game.
                  </Typography>
                </CardContent>
                <CardActions sx={{ justifyContent: "center", mt: "auto" }}>
                  <Button
                    component={Link}
                    to="chess-books"
                    variant="outlined"
                    sx={{
                      color: "#9c27b0",
                      borderColor: "#9c27b0",
                      "&:hover": {
                        borderColor: "#ba68c8",
                        background: "rgba(156, 39, 176, 0.1)",
                      },
                    }}
                    startIcon={<MenuBookIcon />}
                  >
                    Browse
                  </Button>
                </CardActions>
              </GameCard>
            </Grid>
          </Grid>

          {/* Welcome Popup */}
          <Dialog
            open={openPopup}
            onClose={() => setOpenPopup(false)}
            PaperProps={{
              sx: {
                background: "linear-gradient(135deg, #1e1e1e 0%, #121212 100%)",
                borderRadius: "16px",
                border: "1px solid rgba(156, 39, 176, 0.3)",
                boxShadow: "0 0 20px rgba(156, 39, 176, 0.5)",
              },
            }}
          >
            <DialogTitle
              sx={{
                background:
                  "linear-gradient(90deg, rgba(156, 39, 176, 0.2), rgba(33, 150, 243, 0.2))",
                borderBottom: "1px solid rgba(255, 255, 255, 0.1)",
              }}
            >
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <InfoIcon sx={{ color: "#f44336" }} />
                <Typography variant="h6" sx={{ color: "#fff" }}>
                  No Analysis Found
                </Typography>
              </Box>
            </DialogTitle>
            <DialogContent sx={{ mt: 2 }}>
              <Typography sx={{ color: "#b3b3b3" }}>
                It looks like you haven't analyzed any games yet. Please import
                games from the "Import Games" section first!
              </Typography>
            </DialogContent>
            <DialogActions sx={{ p: 2 }}>
              <Button
                onClick={() => setOpenPopup(false)}
                sx={{
                  background:
                    "linear-gradient(45deg, #f44336 30%, #ff5252 90%)",
                  color: "white",
                  "&:hover": {
                    boxShadow: "0 0 10px rgba(244, 67, 54, 0.5)",
                  },
                }}
              >
                Got It
              </Button>
            </DialogActions>
          </Dialog>

          <Box sx={{ mt: 4 }}>
            <Outlet />
          </Box>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default Dashboard;