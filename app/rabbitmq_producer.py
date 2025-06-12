import logging
import asyncio
import aio_pika
import json
import uuid
from app.config import rabbitmq_config # 생성한 설정 파일 import

logger = logging.getLogger(__name__)

_connection = None
_channel = None
_exchange = None

async def get_rabbitmq_connection():
    """
    RabbitMQ 연결을 가져오거나 기존 연결을 반환합니다.
    연결이 없거나 닫혀 있으면 새 연결을 설정합니다.
    """
    global _connection
    if _connection is None or _connection.is_closed:
        logger.info(f"Attempting to connect to RabbitMQ at {rabbitmq_config.RABBITMQ_HOST}:{rabbitmq_config.RABBITMQ_PORT}")
        try:
            _connection = await aio_pika.connect_robust(
                host=rabbitmq_config.RABBITMQ_HOST,
                port=rabbitmq_config.RABBITMQ_PORT,
                login=rabbitmq_config.RABBITMQ_USER,
                password=rabbitmq_config.RABBITMQ_PASSWORD,
                virtualhost=rabbitmq_config.RABBITMQ_VIRTUAL_HOST,
                timeout=10 # 연결 타임아웃 설정 (초 단위)
            )
            logger.info("Successfully connected to RabbitMQ.")
            # 연결이 끊어졌을 때 재연결 로직을 위한 이벤트 핸들러 (선택 사항)
            # _connection.add_close_callback(_on_connection_closed)
            # _connection.add_reconnect_callback(_on_connection_reconnected)
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            _connection = None # 연결 실패 시 None으로 유지
            raise # 연결 실패 시 예외를 다시 발생시켜 호출 측에서 알 수 있도록 함
    return _connection

async def get_rabbitmq_channel():
    """
    RabbitMQ 채널 및 Exchange를 가져오거나 기존 것을 반환합니다.
    채널이 없거나 닫혀 있으면 새 채널을 만들고 Exchange를 선언합니다.
    """
    global _channel, _exchange
    connection = await get_rabbitmq_connection()
    if connection is None:
        logger.error("Cannot get RabbitMQ channel because connection is not established.")
        return None, None # 연결이 없으면 채널도 없음

    if _channel is None or _channel.is_closed:
        try:
            _channel = await connection.channel()
            logger.info("RabbitMQ channel obtained.")
            # Exchange 선언 (없으면 생성, idempotent)
            _exchange = await _channel.declare_exchange(
                name=rabbitmq_config.SERVICE_EXCHANGE_NAME,
                type=aio_pika.ExchangeType(rabbitmq_config.SERVICE_EXCHANGE_TYPE), # aio_pika.ExchangeType 사용
                durable=True # 브로커가 재시작되어도 Exchange 유지
            )
            logger.info(f"RabbitMQ exchange '{rabbitmq_config.SERVICE_EXCHANGE_NAME}' declared.")
        except Exception as e:
            logger.error(f"Failed to get RabbitMQ channel or declare exchange: {e}")
            _channel = None
            _exchange = None
            # 필요시 예외를 다시 발생시킬 수 있음
    return _channel, _exchange

async def publish_message(routing_key: str, message_body: dict):
    """
    지정된 라우팅 키로 메시지를 RabbitMQ Exchange에 발행합니다.
    """
    try:
        channel, exchange = await get_rabbitmq_channel()
        if not channel or not exchange:
            logger.error(f"Cannot publish message to {routing_key} due to missing channel or exchange.")
            return False

        message = aio_pika.Message(
            body=json.dumps(message_body).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT, # 메시지 영속성
            content_type="application/json",
            message_id=str(uuid.uuid4()) # 각 메시지에 고유 ID 부여
        )

        await exchange.publish(message, routing_key=routing_key)
        logger.info(f"Successfully published message to routing key '{routing_key}' (ID: {message.message_id})")
        # logger.debug(f"Message content: {message_body}")
        return True
    except Exception as e:
        logger.error(f"Error publishing message to routing key '{routing_key}': {e}")
        # 필요시 예외 재발생 또는 특정 값 반환
        return False

async def publish_response_message(message_body: dict, exchange_name: str = None, routing_key: str = None):
    """
    응답 메시지를 특정 exchange로 발행합니다 (AI 서버가 BE에 응답할 때 사용).
    """
    try:
        connection = await get_rabbitmq_connection()
        if connection is None:
            logger.error("Cannot publish response message due to missing connection.")
            return False
            
        channel = await connection.channel()
        
        # 응답용 exchange 선언
        response_exchange = await channel.declare_exchange(
            name=exchange_name or rabbitmq_config.QUIZ_RESPONSE_EXCHANGE_NAME,
            type=aio_pika.ExchangeType(rabbitmq_config.QUIZ_RESPONSE_EXCHANGE_TYPE),
            durable=True
        )

        message = aio_pika.Message(
            body=json.dumps(message_body).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            content_type="application/json",
            message_id=str(uuid.uuid4())
        )

        response_routing_key = routing_key or rabbitmq_config.QUIZ_RESPONSE_ROUTING_KEY
        await response_exchange.publish(message, routing_key=response_routing_key)
        
        logger.info(f"Successfully published response message to routing key '{response_routing_key}' (ID: {message.message_id})")
        await channel.close()
        return True
        
    except Exception as e:
        logger.error(f"Error publishing response message: {e}")
        return False

async def close_rabbitmq_connection():
    """
    RabbitMQ 연결 및 채널을 닫습니다.
    """
    global _connection, _channel, _exchange
    if _channel and not _channel.is_closed:
        try:
            await _channel.close()
            logger.info("RabbitMQ channel closed.")
        except Exception as e:
            logger.error(f"Error closing RabbitMQ channel: {e}")
    _channel = None
    _exchange = None # 채널이 닫히면 exchange도 유효하지 않음

    if _connection and not _connection.is_closed:
        try:
            await _connection.close()
            logger.info("RabbitMQ connection closed.")
        except Exception as e:
            logger.error(f"Error closing RabbitMQ connection: {e}")
    _connection = None
