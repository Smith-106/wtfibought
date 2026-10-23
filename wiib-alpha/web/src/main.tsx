import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
// Archivo 用带 wdth 轴的那份（同主站）
import '@fontsource-variable/archivo/wdth.css';
import '@fontsource-variable/jetbrains-mono';
import './index.css';
import './i18n';
import { App } from './App';
import { ToastProvider } from './components/Toasts';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ToastProvider>
      <App />
    </ToastProvider>
  </StrictMode>,
);
