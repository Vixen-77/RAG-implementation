import axios from 'axios';

const api = axios.create({
  baseURL: '/api', // Proxy in Vite will handle this to http://localhost:3000/api
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add a request interceptor to add the JWT token to headers
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => console.error(error)
);

export default api;
