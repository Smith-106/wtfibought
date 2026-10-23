import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import zh from './locales/zh.json';
import en from './locales/en.json';

export type Lang = 'zh' | 'en';

// 与 wiib-web 同一个 key：同域部署时语言跟着走
const LANG_KEY = 'wiib-lang';

void i18n.use(initReactI18next).init({
  resources: { zh: { translation: zh }, en: { translation: en } },
  lng: localStorage.getItem(LANG_KEY) === 'en' ? 'en' : 'zh',
  fallbackLng: 'zh',
  interpolation: { escapeValue: false },
});

const applyHtmlLang = (lang: string) => {
  document.documentElement.lang = lang === 'en' ? 'en' : 'zh-CN';
};
applyHtmlLang(i18n.language);
i18n.on('languageChanged', applyHtmlLang);

export const currentLang = (): Lang => (i18n.language === 'en' ? 'en' : 'zh');

export function setLang(lang: Lang): void {
  localStorage.setItem(LANG_KEY, lang);
  void i18n.changeLanguage(lang);
}

export default i18n;
