import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Box, Typography, TextField, Button, Grid } from '@mui/material';
import GoogleIcon from '@mui/icons-material/Google';
import FacebookIcon from '@mui/icons-material/Facebook';
import ChessLoginImage from '../images/chesslogin.jpg'; // Import the image

function Register() {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [authMessage, setAuthMessage] = useState('');
  const [passwordError, setPasswordError] = useState('');

  const handleRegister = async (e) => {
    e.preventDefault();
    if (password !== confirmPassword) {
      setPasswordError('Passwords do not match');
      return;
    }
    setPasswordError('');
    try {
      const response = await fetch('http://localhost:5000/api/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password }),
      });
      const data = await response.json();
      setAuthMessage(data.message || 'Registration successful!');
    } catch (error) {
      setAuthMessage('Error during registration.');
    }
  };

  const handleGoogleSignIn = () => {
    window.location.href = 'http://localhost:5000/api/auth/google';
  };

  const handleFacebookSignIn = () => {
    window.location.href = 'http://localhost:5000/api/auth/facebook';
  };

  return (
    <Box sx={{ minHeight: '100vh', display: 'flex', flexDirection: { xs: 'column', md: 'row' } }}>
      {/* Left Side: Graphics Section (65%) */}
      <Box
        sx={{
          width: { xs: '100%', md: '65%' },
          minHeight: { xs: '30vh', md: '100vh' },
          bgcolor: '#cbeaf6', // Light cyan
          position: 'relative',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          color: '#a1d8ef', // Darker cyan for text
          overflow: 'hidden', // Clip overflow
        }}
      >
        <Typography variant="h3" sx={{ fontFamily: 'Georgia, serif', zIndex: 1, color: '#000' }}>
          Chess Graphics
        </Typography>
        {/* Chessboard Image */}
        <Box
          sx={{
            position: 'absolute',
            right: '-20%', // Adjusted to show more of the image
            top: '50%',
            transform: 'translateY(-50%) rotate(45deg)', // Rotate 45 degrees
            width: '80%', // Bigger to show more
            height: '80%', // Bigger to show more
            backgroundImage: `url(${ChessLoginImage})`, // Use imported image
            backgroundSize: 'cover',
            backgroundPosition: 'center',
          }}
        />
      </Box>

      {/* Right Side: Form Section (35%) */}
      <Box
        sx={{
          width: { xs: '100%', md: '35%' },
          minHeight: { xs: '70vh', md: '100vh' },
          bgcolor: '#a1d8ef', // Darker cyan
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          p: 3,
        }}
      >
        <Typography variant="h5" gutterBottom sx={{ color: '#cbeaf6', fontFamily: 'Georgia, serif' }}>
          Join the Chess Arena
        </Typography>
        <Box component="form" onSubmit={handleRegister} sx={{ width: '100%', maxWidth: 300 }}>
          <TextField
            label="Username"
            variant="outlined"
            fullWidth
            margin="normal"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            sx={{ input: { color: '#cbeaf6' }, label: { color: '#cbeaf6' } }}
          />
          <TextField
            label="Email"
            type="email"
            variant="outlined"
            fullWidth
            margin="normal"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            sx={{ input: { color: '#cbeaf6' }, label: { color: '#cbeaf6' } }}
          />
          <TextField
            label="Password"
            type="password"
            variant="outlined"
            fullWidth
            margin="normal"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            sx={{ input: { color: '#cbeaf6' }, label: { color: '#cbeaf6' } }}
          />
          <TextField
            label="Confirm Password"
            type="password"
            variant="outlined"
            fullWidth
            margin="normal"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            required
            error={!!passwordError}
            helperText={passwordError}
            sx={{ input: { color: '#cbeaf6' }, label: { color: '#cbeaf6' } }}
          />
          <Button
            type="submit"
            variant="contained"
            fullWidth
            sx={{ mt: 2, bgcolor: '#cbeaf6', color: '#a1d8ef', '&:hover': { bgcolor: '#b3e0f2' } }}
          >
            Register
          </Button>
        </Box>
        <Typography variant="body2" sx={{ mt: 2, color: '#cbeaf6' }}>
          {authMessage}
        </Typography>
        <Grid container spacing={2} sx={{ mt: 2, width: '100%', maxWidth: 300 }}>
          <Grid item xs={6}>
            <Button
              variant="outlined"
              startIcon={<GoogleIcon />}
              fullWidth
              onClick={handleGoogleSignIn}
              sx={{ borderColor: '#cbeaf6', color: '#cbeaf6' }}
            >
              Google
            </Button>
          </Grid>
          <Grid item xs={6}>
            <Button
              variant="outlined"
              startIcon={<FacebookIcon />}
              fullWidth
              onClick={handleFacebookSignIn}
              sx={{ borderColor: '#cbeaf6', color: '#cbeaf6' }}
            >
              Facebook
            </Button>
          </Grid>
        </Grid>
        <Typography variant="body2" sx={{ mt: 2, color: '#cbeaf6' }}>
          Already have an account?{' '}
          <Link to="/login" style={{ color: '#cbeaf6' }}>Login here</Link>
        </Typography>
      </Box>
    </Box>
  );
}

export default Register;