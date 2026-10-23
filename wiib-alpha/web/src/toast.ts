import { createContext, useCallback, useContext } from 'react';
import { ApiError } from './api';

export type ToastKind = 'error' | 'success';

export const ToastContext = createContext<(message: string, kind?: ToastKind) => void>(() => {});

/** 动手的接口拿到 401/403 交给页面：切回只读并提示 */
export const AuthErrorContext = createContext<(code: 401 | 403) => void>(() => {});

export const useToast = () => useContext(ToastContext);

/** 接口报错弹出来；401/403 走 AuthErrorContext */
export function useFail() {
  const toast = useToast();
  const onAuthError = useContext(AuthErrorContext);
  return useCallback((err: unknown) => {
    if (err instanceof ApiError && (err.code === 401 || err.code === 403)) onAuthError(err.code);
    else toast(err instanceof Error ? err.message : String(err), 'error');
  }, [toast, onAuthError]);
}
