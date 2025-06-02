import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from typing import List, Dict, Optional
import torch
import logging
import numpy as np
import re
from app.vector_db.utils import embed_texts, clean_keyword_phrase, get_keyword_model
from app.vector_db.chroma_client import (
    get_all_question_documents_with_vectors,
    get_all_quiz_documents_with_vectors
)
from app.vector_db.config import RAG_TOP_K, SIM_THRESHOLD, RAG_DIVERSITY

def extract_keywords_fallback(text: str, fallback_n: int = 3) -> List[str]:
    """텍스트에서 키워드를 추출하는 함수"""
    try:
        if len(text.strip().split()) <= 2:
            return [clean_keyword_phrase(text)]
        
        if "," in text:
            return [clean_keyword_phrase(k) for k in text.split(",") if k.strip()]

        kw_model = get_keyword_model()
        dynamic_top_n = min(fallback_n, max(1, len(text.split()) // 2))

        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words=None,
            top_n=dynamic_top_n,
            use_mmr=True,
            diversity=RAG_DIVERSITY
        )

        phrases = [clean_keyword_phrase(k[0]) for k in keywords if k[1] >= 0.5]
        phrases = [clean_keyword_phrase(p) for p in phrases]  

        return phrases if phrases else [clean_keyword_phrase(text)]
        
    except Exception as e:
        logging.warning(f"키워드 추출 실패: {e}")
        return [clean_keyword_phrase(text)]


def safe_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """코사인 유사도 계산"""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


def question_rag_retriever(
    main_question: str,
    keyword: Optional[str] = None,
    top_k: int = 4,
    sim_threshold: float = 0.6,
    base_question_weight: float = 0.3,
    base_keyword_weight: float = 0.7
) -> List[Dict[str, float]]:
    """꼬리질문 생성용 RAG 검색기 (키워드 기반 검색 포함)"""
    try:
        question_keywords = extract_keywords_fallback(main_question)
        question_keyword = ", ".join(question_keywords)
        
        main_embeds = embed_texts([main_question],enrich_type='question')
        kw_embeds = embed_texts(question_keywords,enrich_type='question')

        q_vec_main = np.array(main_embeds[0])
        q_vec_kw = np.mean(np.array(kw_embeds), axis=0)
        q_vec = 0.6 * q_vec_main + 0.4 * q_vec_kw

        if keyword:
            keyword_phrases = extract_keywords_fallback(keyword)
            auto_keyword = ", ".join(keyword_phrases)

            if keyword_phrases:
                k_vec = np.mean(embed_texts(keyword_phrases,enrich_type='question'), axis=0)
                question_weight, keyword_weight = 0.6, 0.4
            else:
                k_vec = None
                question_weight, keyword_weight = 1.0, 0.0

        else:
            auto_keyword = ""
            k_vec = None
            question_weight, keyword_weight = 1.0, 0.0


        all_docs = get_all_question_documents_with_vectors()
        results = []

        for _, doc_text, doc_vec in all_docs:

            q_sim = safe_cosine_similarity(np.array(q_vec), np.array(doc_vec))
            k_sim = safe_cosine_similarity(np.array(k_vec), np.array(doc_vec)) if k_vec is not None else 0.0

            final_sim = question_weight * q_sim + keyword_weight * k_sim

            if auto_keyword:
                keyword_list = set([kw.strip().lower() for kw in auto_keyword.split(",")])
                keyword_bonus = sum(
                    1 for kw in keyword_list
                    if re.search(rf"\b{re.escape(kw)}\b", doc_text.lower())
                ) * 0.05
                final_sim += min(keyword_bonus, 0.15)


            if final_sim >= sim_threshold:
                results.append({
                    "question": doc_text,
                    "similarity": round(final_sim, 4),
                    "question_keyword": question_keyword,
                    "auto_keyword": auto_keyword
                })

        sorted_results = sorted(results, key=lambda x: x["similarity"], reverse=True)
        return {
            "results": sorted_results[:top_k],
            "question_keyword": question_keyword,
            "auto_keyword": auto_keyword
        }


    except Exception as e:
        logging.warning(f"RAG 검색 실패: {e}")
        return []

def quiz_rag_retriever(
    questions: List[str],
    top_k: int = 3,
    sim_threshold: float = 0.7
) -> List[Dict[str, any]]:
    """
    여러 개의 질문에 대해 RAG 검색을 수행하여 각각의 결과를 반환
    """
    if not questions:
        logging.warning("검색할 질문이 없습니다.")
        return []
    
    results = []
    
    try:
        logging.info(f"퀴즈 RAG 검색 시작: {len(questions)}개 질문 처리")
        
        question_embeds = embed_texts(questions, enrich_type='quiz')
        
        all_docs = get_all_quiz_documents_with_vectors()
        
        if not all_docs:
            logging.warning("검색할 문서가 없습니다.")
            return [{"질문": q, "결과": []} for q in questions]
        
        for idx, question in enumerate(questions):
            logging.debug(f"질문 {idx+1}/{len(questions)} 처리 중: '{question[:50]}...'")
            
            try:
                question_vec = np.array(question_embeds[idx])
                matched_docs = []
                
                for doc_id, doc_text, doc_vec in all_docs:
                    similarity = safe_cosine_similarity(question_vec, np.array(doc_vec))
                    
                    if similarity >= sim_threshold:
                        matched_docs.append({
                            "document_id": doc_id,
                            "content": doc_text,
                            "similarity": round(similarity, 4)
                        })
                
                sorted_docs = sorted(matched_docs, key=lambda x: x["similarity"], reverse=True)[:top_k]
                
                question_result = {
                    "result": sorted_docs
                }
                
                results.append(question_result)
                
                logging.debug(f"질문 '{question[:30]}...' 검색 완료: {len(sorted_docs)}개 문서 매칭")
                
            except Exception as e:
                logging.error(f"질문 '{question[:30]}...' 검색 실패: {e}")
                results.append({
                    "result": []
                })
        
        logging.info(f"퀴즈 RAG 검색 완료: {len(results)}개 결과 반환")
        
    except Exception as e:
        logging.error(f"퀴즈 RAG 배치 검색 실패: {e}")
        results = []
        for question in questions:
            results.append({
                "result": []
            })
    
    return results