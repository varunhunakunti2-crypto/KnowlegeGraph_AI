"""Database base and lifecycle management."""
from typing import Optional
from neo4j import GraphDatabase, Driver
import psycopg
from app.core.config import settings
from app.core.logging import logger


class DatabaseManager:
    """Manages database connections with graceful startup/shutdown."""
    
    def __init__(self):
        self.neo4j_driver: Optional[Driver] = None
        self.pg_conn_pool = None
    
    async def startup(self):
        """Initialize database connections."""
        logger.info("Initializing database connections...")
        # Neo4j
        try:
            self.neo4j_driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            self.neo4j_driver.verify_connectivity()
            logger.info("Neo4j connection established")
        except Exception as e:
            logger.warning(f"Neo4j connection failed: {type(e).__name__}")
        
        # PostgreSQL
        try:
            # Simple connection test
            conn = psycopg.connect(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                dbname=settings.POSTGRES_DB,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                connect_timeout=5,
            )
            conn.close()
            logger.info("PostgreSQL connection established")
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {type(e).__name__}")
    
    async def shutdown(self):
        """Close database connections gracefully."""
        logger.info("Closing database connections...")
        if self.neo4j_driver:
            try:
                self.neo4j_driver.close()
                logger.info("Neo4j connection closed")
            except Exception as e:
                logger.warning(f"Error closing Neo4j: {type(e).__name__}")


db_manager = DatabaseManager()
