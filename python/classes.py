from loguru import logger
import os
from email import policy
from email.parser import BytesParser
import extract_msg
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Any, Tuple, Optional
import uuid

URL_REGEX = re.compile(
    r"""(?i)\b((?:https?://|www\.)[^\s<>'"]+)"""
)

class MessageParser:
    def __init__(self):
        self.messages = []

    def _extract_links_from_text(self, text: str) -> List[str]:
        """
        Извлечение ссылок из простого текста.
        """
        logger.info("Выполнение скрипта по извлечению ссылок из текста")
        try:
            if not text:
                logger.warning('переменная text - пустая, возвращен пустой массив')
                return []
            logger.info("Скрипт по извлечению ссылок из текста выполнен")
            return list(set(URL_REGEX.findall(text)))
        except Exception as e:
            logger.error(f"Произошла ошибка - {e}")

    def _clean_html(self, html: str) -> str:
        """
        Удаление html-тегов, скриптов, стилей, служебного мусора.
        """
        logger.info("Выполнение скрипта по очистке HTML")
        try:
            soup = BeautifulSoup(html, "html.parser")

            # Удаляем скрипты и стили
            for tag in soup(["script", "style"]):
                tag.decompose()

            text = soup.get_text(separator=" ")
            logger.info("Скрипт по очистке HTML выполнен")
            return self._normalize_whitespace(text)
        except Exception as e:
            logger.error(f"Произошла ошибка - {e}")

    def _normalize_whitespace(self, text: str) -> str:
        """
        Нормализация пробелов и переносов строк.
        """
        logger.info("Выполнение скрипта по нормализации пробелов и переносов строк")
        try:
            if not text:
                logger.warning('переменная text - пустая, возвращен пустая строка')
                return ""
            # заменяем любые последовательности пробелов/переносов одним пробелом
            text = re.sub(r"\s+", " ", text)
            logger.info("Скрипт по нормализации пробелов и переносов строк выполнен")
            return text.strip()
        except Exception as e:
            logger.error(f"Произошла ошибка - {e}")

    def _parse_eml_bytes(self, data: bytes) -> Tuple[str, str, str, List[str], List[str]]:
        """
        Парсинг .eml из байтов.
        Возвращает:
          subject, plain_text, html_text, attachments_text_list, links_list
        """
        logger.info("Выполнение скрипта по парсингу писем формата eml")
        msg = BytesParser(policy=policy.default).parsebytes(data)

        subject = msg.get("subject", "") or ""
        plain_parts: List[str] = []
        html_parts: List[str] = []
        attachments_text: List[str] = []
        links: List[str] = []

        if msg.is_multipart():
            for part in msg.walk():
                content_disposition = part.get_content_disposition()
                content_type = part.get_content_type()

                # Тело письма: text/plain и text/html без content-disposition=attachment
                if content_disposition is None:
                    if content_type == "text/plain":
                        try:
                            payload = part.get_content()
                        except Exception as e:
                            logger.error(f"Произошла ошибка - {e}")
                            payload = part.get_payload(decode=True) or b""
                            payload = payload.decode(errors="ignore")
                        plain_parts.append(str(payload))
                        links.extend(self._extract_links_from_text(str(payload)))
                    elif content_type == "text/html":
                        try:
                            payload = part.get_content()
                        except Exception as e:
                            logger.error(f"Произошла ошибка - {e}")
                            payload = part.get_payload(decode=True) or b""
                            payload = payload.decode(errors="ignore")
                        html_parts.append(str(payload))
                        # ссылки потом извлечём из html
                else:
                    # Вложения
                    if content_disposition == "attachment":
                        try:
                            filename = part.get_filename() or ""
                            maintype = part.get_content_maintype()
                            subtype = part.get_content_subtype()

                            payload = part.get_payload(decode=True) or b""
                            text_content = ""

                            # текстовые вложения
                            if maintype == "text" or subtype in {"plain", "html", "csv"}:
                                text_content = payload.decode(errors="ignore")

                            if text_content:
                                attachments_text.append(text_content)
                                links.extend(self._extract_links_from_text(text_content))
                        except Exception:
                            continue
        else:
            content_type = msg.get_content_type()
            if content_type == "text/plain":
                payload = msg.get_content()
                plain_parts.append(str(payload))
                links.extend(self._extract_links_from_text(str(payload)))
            elif content_type == "text/html":
                payload = msg.get_content()
                html_parts.append(str(payload))

        html_text = "\n\n".join(html_parts)
        plain_text = "\n\n".join(plain_parts)

        # ссылки из html
        for html_part in html_parts:
            soup = BeautifulSoup(html_part, "html.parser")
            for a in soup.find_all("a", href=True):
                links.append(a["href"])

        # убираем дубликаты ссылок
        links = list(set(links))
        logger.info("Cкрипта по парсингу писем формата eml выполнен")
        return subject, plain_text, html_text, attachments_text, links

    def _parse_msg_file(self, path: str) -> Tuple[str, str, str, List[str], List[str]]:
        """
        Парсинг .msg c помощью extract_msg.
        Возвращает:
          subject, plain_text, html_text, attachments_text_list, links_list
        """
        logger.info("Выполнение скрипта по парсингу писем формата msg")
        msg = extract_msg.Message(path)

        subject = msg.subject or ""
        plain_text = msg.body or ""
        html_text = msg.htmlBody or ""

        attachments_text: List[str] = []
        links: List[str] = []

        # вложения
        for att in msg.attachments:
            try:
                data = att.data
                # попробуем декодировать как текст
                try:
                    text_content = data.decode(errors="ignore")
                except AttributeError as e:
                    logger.error(f"Произошла ошибка - {e}")
                    # иногда data уже строка
                    text_content = str(data)

                if text_content:
                    attachments_text.append(text_content)
                    links.extend(self._extract_links_from_text(text_content))
            except Exception:
                continue

        # ссылки из plain_text
        links.extend(self._extract_links_from_text(plain_text))

        # ссылки из html_text
        if html_text:
            soup = BeautifulSoup(html_text, "html.parser")
            for a in soup.find_all("a", href=True):
                links.append(a["href"])

        links = list(set(links))
        logger.info("Cкрипта по парсингу писем формата msg выполнен")
        return subject, plain_text, html_text, attachments_text, links

    def parse_email_file(self, path: str) -> Dict[str, Any]:
        """
        Универсальный парсер для .eml и .msg.
        Возвращает словарь:
        {
          "id": str,
          "subject": str,
          "body": str,              # очищенный текст письма
          "attachments_text": str,  # очищенный текст вложений (объединённый)
          "links": List[str],
          "full_text": str          # subject + body + attachments_text
        }
        """
        logger.info("Выполнение скрипта по парсингу письма")
        ext = os.path.splitext(path)[1].lower()

        if ext == ".eml":
            with open(path, "rb") as f:
                data = f.read()
            subject, plain_text, html_text, attachments_text_list, links = self._parse_eml_bytes(data)
        elif ext == ".msg":
            subject, plain_text, html_text, attachments_text_list, links = self._parse_msg_file(path)
        else:
            raise ValueError(f"Сервис не поддерживает этот формат {ext}")
            logger.error(f"Сервис не поддерживает этот формат {ext}")

        # Ручная предобработка

        # чистим html
        html_clean = self._clean_html(html_text) if html_text else ""

        # plain text тоже нормализуем
        plain_clean = self._normalize_whitespace(plain_text)

        # объединяем body
        if plain_clean and html_clean:
            body = f"{plain_clean}\n\n{html_clean}"
        elif plain_clean:
            body = plain_clean
        else:
            body = html_clean

        # 4. текст вложений
        attachments_clean_list = [self._normalize_whitespace(self._clean_html(t)) for t in attachments_text_list]
        attachments_text = "\n\n".join([t for t in attachments_clean_list if t])

        # 5. итоговый full_text
        subject_clean = self._normalize_whitespace(subject)
        parts = [subject_clean, body, attachments_text]
        full_text = "\n\n".join([p for p in parts if p])

        # 6. id письма из uuid4
        email_id = str(uuid.uuid4())

        result = {
            "id": email_id,
            "subject": subject_clean,
            "body": body,
            "attachments_text": attachments_text,
            "links": links,
            "full_text": full_text,
        }
        logger.info("Cкрипта по парсингу письма выполнен")
        return result


from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingService:
    def __init__(self, model_name="BAAI/bge-m3"):
        logger.info(f"Инициализация EmbeddingService с моделью: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            logger.success(f"Модель {model_name} успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели {model_name}: {str(e)}")
            raise

    def embed(self, texts):
        logger.debug(f"Создание эмбеддингов для {len(texts)} текстов")
        if texts and logger.level("DEBUG").no <= logger._core.min_level:
            sample_text = texts[0][:50] + "..." if len(texts[0]) > 50 else texts[0]
            logger.debug(f"Пример текста: '{sample_text}'")

        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            logger.debug(f"Эмбеддинги успешно созданы, shape: {embeddings.shape}")
            if logger.level("DEBUG").no <= logger._core.min_level:
                logger.debug(f"Норма эмбеддинга: min={np.min(np.linalg.norm(embeddings, axis=1)):.4f}, "f"max={np.max(np.linalg.norm(embeddings, axis=1)):.4f}")
            return embeddings
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддингов: {str(e)}")
            raise

from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class Category:
    name: str
    examples: Optional[List[str]] = None

class MailClassyfire:
    def _calculate_adaptive_threshold(self, scores: np.ndarray) -> float:
        """
        Вычисляет адаптивный порог на основе распределения scores.
        Использует среднее значение между максимальным и средним score.
        """
        if len(scores) == 0:
            return 0.3
        
        max_score = scores.max()
        mean_score = scores.mean()
        std_score = scores.std()
        
        # Адаптивный порог: среднее между max и mean, с учетом стандартного отклонения
        adaptive_threshold = mean_score + 0.5 * (max_score - mean_score)
        
        # Ограничиваем диапазоном [0.2, 0.4] для low confidence
        adaptive_threshold = max(0.2, min(0.4, adaptive_threshold))
        
        logger.debug(f"Адаптивный порог: {adaptive_threshold:.4f} (max={max_score:.4f}, mean={mean_score:.4f}, std={std_score:.4f})")
        return adaptive_threshold
    
    def _get_confidence_level(self, score: float, threshold: float) -> str:
        """
        Определяет уровень уверенности на основе score и порога.
        """
        if score >= threshold:
            return "high"
        elif score >= threshold * 0.7:  # 70% от порога
            return "medium"
        elif score >= 0.2:
            return "low"
        else:
            return "very_low"
    
    def classify_email(
            self,
            email_text: str,
            categories: List[Any],
            embedder: EmbeddingService,
            file_name : str,
            threshold: Optional[float] = None,
            top_n: int = 3
    ) -> Dict[str, Any]:
        logger.info(f"Начинаем классификацию email (длина: {len(email_text)} символов)")
        logger.debug(f"Параметры классификации: threshold={threshold}, top_n={top_n}")

        try:
            logger.debug("Создаем эмбеддинг для email...")
            email_vec = embedder.embed([email_text])[0]
            logger.debug(f"Эмбеддинг email создан, размерность: {email_vec.shape}")
            logger.debug("Строим векторы категорий...")
            category_vectors = self.build_category_vectors(categories, embedder)

            if not category_vectors:
                error_msg = "Не удалось построить векторы категорий: проверьте, что Category.name непустые строки."
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(f"Успешно построены векторы для {len(category_vectors)} категорий")
            logger.debug("Подготавливаем матрицу векторов категорий...")
            names = list(category_vectors.keys())
            vectors = np.vstack([category_vectors[n] for n in names])
            logger.debug(f"Матрица векторов категорий создана, shape: {vectors.shape}")
            logger.debug("Вычисляем косинусное сходство...")
            scores = cosine_similarity(email_vec.reshape(1, -1), vectors)[0]
            if logger.level("DEBUG").no <= logger._core.min_level:
                logger.debug(
                    f"Статистика scores: min={scores.min():.4f}, "f"max={scores.max():.4f}, mean={scores.mean():.4f}")
            
            # Сортируем по убыванию score
            ranked = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
            
            # Используем адаптивный порог, если не указан явно
            if threshold is None:
                threshold = self._calculate_adaptive_threshold(scores)
            
            # Берем топ-3 категории
            top_3 = ranked[:min(3, len(ranked))]
            top_scores = [
                {
                    "category": n, 
                    "score": float(s),
                    "rank": idx + 1,
                    "confidence": self._get_confidence_level(float(s), threshold)
                } 
                for idx, (n, s) in enumerate(top_3)
            ]

            best_cat, best_score = ranked[0]
            confidence_level = self._get_confidence_level(float(best_score), threshold)
            is_undefined = best_score < threshold

            # Вычисляем метрику confidence (нормализованный score от 0 до 1)
            # Используем сигмоиду для нормализации
            confidence_metric = 1 / (1 + np.exp(-5 * (best_score - threshold)))
            
            # Альтернативная метрика: процент от максимально возможного
            max_possible_score = 1.0
            confidence_percentage = (best_score / max_possible_score) * 100

            result = {
                "predicted_category": "Не определена" if is_undefined else best_cat,
                "is_undefined": bool(is_undefined),
                "best_score": float(best_score),
                "confidence_level": confidence_level,
                "confidence_metric": float(confidence_metric),
                "confidence_percentage": float(confidence_percentage),
                "threshold_used": float(threshold),
                "top_3_categories": top_scores,
                "scores": top_scores,  # Для обратной совместимости
                "file_name": file_name
            }

            if is_undefined:
                logger.warning(
                    f"Категория не определена. Лучший результат: '{best_cat}' "
                    f"с score={best_score:.4f} (ниже порога {threshold:.4f}), confidence={confidence_level}"
                )
            else:
                logger.info(
                    f"Определена категория: '{best_cat}' с score={best_score:.4f} "
                    f"(порог: {threshold:.4f}), confidence={confidence_level}"
                )

            logger.debug(f"Топ-3 категорий: {top_scores}")
            logger.debug(f"Лучшая категория: {best_cat}, score: {best_score:.4f}, confidence: {confidence_metric:.4f}")
            logger.success(f"Классификация завершена успешно")
            return result
        except ValueError as ve:
            logger.error(f"Ошибка валидации при классификации: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Неожиданная ошибка при классификации email: {str(e)}")
            logger.exception("Трассировка ошибки:")
            raise

    def build_category_vectors(self, categories: List[Category], embedder: EmbeddingService) -> Dict[str, np.ndarray]:
        logger.info(f"Начинаем построение векторов для {len(categories)} категорий")

        category_vectors = {}
        processed_count = 0
        skipped_count = 0

        for idx, cat in enumerate(categories, 1):
            name = (cat.name or "").strip()
            if not name:
                logger.warning(f"Категория #{idx} пропущена - пустое имя")
                skipped_count += 1
                continue

            logger.debug(f"Обрабатываем категорию #{idx}: '{name}'")

            texts = [name]
            if cat.examples:
                valid_examples = [e.strip() for e in cat.examples if isinstance(e, str) and e.strip()]
                texts.extend(valid_examples)
                logger.debug(f"  Добавлено {len(valid_examples)} примеров для категории '{name}'")
            else:
                logger.debug(f"  Примеры для категории '{name}' отсутствуют, используем только название")

            try:
                vecs = embedder.embed(texts)
                logger.debug(f"  Получено {vecs.shape[0]} эмбеддингов размерностью {vecs.shape[1]}")
                proto = np.mean(vecs, axis=0)
                logger.debug(f"  Вычислено среднее эмбеддингов")
                proto = proto / (np.linalg.norm(proto) + 1e-12)
                category_vectors[name] = proto
                processed_count += 1
                logger.debug(f"  Успешно создан вектор для категории '{name}' (норма: {np.linalg.norm(proto):.4f})")

            except Exception as e:
                logger.error(f"  Ошибка при обработке категории '{name}': {str(e)}")
                skipped_count += 1
                continue

        logger.info(
            f"Построение векторов завершено: "f"обработано {processed_count}, пропущено {skipped_count}, всего векторов: {len(category_vectors)}")
        if len(category_vectors) == 0:
            logger.error("Не удалось построить ни одного вектора категорий!")

        return category_vectors