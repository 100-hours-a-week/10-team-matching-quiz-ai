#!/usr/bin/env python3
"""
꼬리질문 생성 기능 커버리지 분석 스크립트
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
import sys


def analyze_coverage_xml():
    """XML 커버리지 리포트 분석"""
    coverage_file = Path("coverage.xml")
    
    if not coverage_file.exists():
        print("❌ coverage.xml 파일이 없습니다. 먼저 테스트를 실행하세요.")
        return None
    
    tree = ET.parse(coverage_file)
    root = tree.getroot()
    
    # 전체 커버리지 정보
    total_lines = int(root.attrib.get('lines-valid', 0))
    covered_lines = int(root.attrib.get('lines-covered', 0))
    total_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
    
    print(f"📊 전체 커버리지: {total_coverage:.1f}% ({covered_lines}/{total_lines} lines)")
    print()
    
    # 모듈별 커버리지 분석
    modules = {}
    for package in root.findall('.//package'):
        package_name = package.attrib.get('name', '')
        
        for class_elem in package.findall('.//class'):
            filename = class_elem.attrib.get('filename', '')
            if 'question_generator' in filename:
                lines_valid = int(class_elem.attrib.get('lines-valid', 0))
                lines_covered = int(class_elem.attrib.get('lines-covered', 0))
                coverage = (lines_covered / lines_valid * 100) if lines_valid > 0 else 0
                
                modules[filename] = {
                    'lines_valid': lines_valid,
                    'lines_covered': lines_covered,
                    'coverage': coverage
                }
    
    # 꼬리질문 생성 관련 모듈 커버리지 출력
    print("🎯 꼬리질문 생성 모듈별 커버리지:")
    print("-" * 60)
    
    question_generator_modules = [
        'question_generator_api.py',
        'question_generator_model.py', 
        'question_generator_schema.py',
        'question_generator_config.py',
        'question_generator_parser.py'
    ]
    
    for module in question_generator_modules:
        found = False
        for filename, data in modules.items():
            if module in filename:
                print(f"📁 {module:30} {data['coverage']:6.1f}% ({data['lines_covered']:3d}/{data['lines_valid']:3d})")
                found = True
                break
        
        if not found:
            print(f"📁 {module:30}   0.0% (not tested)")
    
    return total_coverage


def analyze_coverage_priorities():
    """커버리지 개선 우선순위 분석"""
    print("\n🔍 커버리지 개선 권장사항:")
    print("-" * 60)
    
    priorities = [
        {
            "module": "question_generator_api.py",
            "priority": "높음",
            "reason": "핵심 API 엔드포인트, 비즈니스 로직",
            "target": "90%+",
            "focus": ["perform_rag_search", "generate_questions_with_fallback", "prepare_context"]
        },
        {
            "module": "question_generator_model.py", 
            "priority": "높음",
            "reason": "vLLM/OpenAI 모델 통합",
            "target": "80%+",
            "focus": ["call_llm", "call_openai_api", "initialize_llm"]
        },
        {
            "module": "question_generator_schema.py",
            "priority": "중간",
            "reason": "데이터 검증 및 스키마",
            "target": "95%+", 
            "focus": ["FollowupRequest", "FollowupResponse"]
        },
        {
            "module": "question_generator_parser.py",
            "priority": "높음",
            "reason": "질문 파싱 로직",
            "target": "90%+",
            "focus": ["parse_questions"]
        },
        {
            "module": "question_generator_config.py",
            "priority": "낮음", 
            "reason": "설정 파일",
            "target": "70%+",
            "focus": ["환경변수 처리"]
        }
    ]
    
    for i, item in enumerate(priorities, 1):
        print(f"{i}. {item['module']} (우선순위: {item['priority']})")
        print(f"   목표: {item['target']} | 이유: {item['reason']}")
        print(f"   집중영역: {', '.join(item['focus'])}")
        print()


def generate_coverage_report():
    """커버리지 리포트 생성"""
    print("=" * 70)
    print("🧪 꼬리질문 생성 기능 테스트 커버리지 분석")
    print("=" * 70)
    
    coverage = analyze_coverage_xml()
    
    if coverage is not None:
        print(f"\n📈 현재 커버리지 수준:")
        if coverage >= 80:
            print(f"✅ 우수 ({coverage:.1f}%) - 추가 최적화 권장")
        elif coverage >= 60:
            print(f"🟡 양호 ({coverage:.1f}%) - 일부 개선 필요")
        elif coverage >= 40:
            print(f"🟠 보통 ({coverage:.1f}%) - 상당한 개선 필요")
        else:
            print(f"🔴 부족 ({coverage:.1f}%) - 대폭 개선 필요")
    
    analyze_coverage_priorities()
    
    print("💡 테스트 실행 명령어:")
    print("   ./run_tests.sh")
    print("   또는")
    print("   pytest --cov=app --cov-report=html --cov-report=term-missing")
    print()
    print("📋 HTML 상세 리포트: htmlcov/index.html")


if __name__ == "__main__":
    generate_coverage_report()
