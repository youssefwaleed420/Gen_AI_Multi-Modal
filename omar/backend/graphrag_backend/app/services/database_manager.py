import psycopg2
import psycopg2.extras
import uuid
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime

# Database config - Update with your actual credentials
DB_CONFIG = {
    "host": "aws-0-us-west-1.pooler.supabase.com",
    "port": 6543,
    "database": "postgres",
    "user": "postgres.jsgdmwbhzvwdhardoimb",
    "password": "omernasser123"
}

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DatabaseManager:
    """Enhanced PostgreSQL database operations for advanced chat history and analytics"""
    
    def __init__(self):
        self.connection = None
        self.connect()
        self.setup_tables()
    
    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(**DB_CONFIG)
            logger.info("✅ Successfully connected to PostgreSQL!")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {str(e)}")
            self.connection = None
    
    def column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a table"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s AND column_name = %s;
            """, (table_name, column_name))
            return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"❌ Error checking column existence: {str(e)}")
            return False
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name = %s;
            """, (table_name,))
            return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"❌ Error checking table existence: {str(e)}")
            return False
    
    def migrate_schema(self):
        """Migrate existing schema to new enhanced schema"""
        if not self.connection:
            return
            
        try:
            cursor = self.connection.cursor()
            
            # Add new columns to existing chats table if they don't exist
            chat_columns_to_add = [
                ("user_id", "TEXT DEFAULT 'default_user'"),
                ("is_active", "BOOLEAN DEFAULT true"),
                ("total_messages", "INTEGER DEFAULT 0"),
                ("avg_response_time", "FLOAT DEFAULT 0.0"),
                ("knowledge_domains", "JSONB DEFAULT '[]'"),
                ("complexity_score", "FLOAT DEFAULT 0.0")
            ]
            
            for column_name, column_def in chat_columns_to_add:
                if not self.column_exists('chats', column_name):
                    try:
                        cursor.execute(f"ALTER TABLE chats ADD COLUMN {column_name} {column_def};")
                        logger.info(f"✅ Added column {column_name} to chats table")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not add column {column_name}: {str(e)}")
            
            # Add new columns to existing messages table if they don't exist
            message_columns_to_add = [
                ("metadata", "JSONB DEFAULT '{}'"),
                ("context_used", "INTEGER DEFAULT 0"),
                ("graph_context_used", "INTEGER DEFAULT 0"),
                ("voyage_embedding_used", "BOOLEAN DEFAULT false"),
                ("search_strategy", "TEXT DEFAULT 'hybrid'"),
                ("relevance_scores", "JSONB DEFAULT '{}'"),
                ("extracted_entities", "JSONB DEFAULT '[]'"),
                ("graph_paths", "JSONB DEFAULT '[]'"),
                ("response_confidence", "FLOAT DEFAULT 0.0"),
                ("processing_time", "FLOAT DEFAULT 0.0")
            ]
            
            for column_name, column_def in message_columns_to_add:
                if not self.column_exists('messages', column_name):
                    try:
                        cursor.execute(f"ALTER TABLE messages ADD COLUMN {column_name} {column_def};")
                        logger.info(f"✅ Added column {column_name} to messages table")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not add column {column_name}: {str(e)}")
            
            self.connection.commit()
            logger.info("✅ Schema migration completed")
            
        except Exception as e:
            logger.error(f"❌ Failed to migrate schema: {str(e)}")
            if self.connection:
                self.connection.rollback()
    
    def setup_tables(self):
        """Create enhanced tables for GraphRAG analytics"""
        if not self.connection:
            return
            
        try:
            cursor = self.connection.cursor()
            
            # Check if tables exist and migrate if necessary
            if self.table_exists('chats') or self.table_exists('messages'):
                logger.info("🔄 Existing tables found, performing migration...")
                self.migrate_schema()
            
            # Enhanced chats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT DEFAULT 'default_user',
                    is_active BOOLEAN DEFAULT true,
                    total_messages INTEGER DEFAULT 0,
                    avg_response_time FLOAT DEFAULT 0.0,
                    knowledge_domains JSONB DEFAULT '[]',
                    complexity_score FLOAT DEFAULT 0.0
                );
            """)
            
            # Enhanced messages table with GraphRAG analytics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    chat_id UUID REFERENCES chats(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB DEFAULT '{}',
                    context_used INTEGER DEFAULT 0,
                    graph_context_used INTEGER DEFAULT 0,
                    voyage_embedding_used BOOLEAN DEFAULT false,
                    search_strategy TEXT DEFAULT 'hybrid',
                    relevance_scores JSONB DEFAULT '{}',
                    extracted_entities JSONB DEFAULT '[]',
                    graph_paths JSONB DEFAULT '[]',
                    response_confidence FLOAT DEFAULT 0.0,
                    processing_time FLOAT DEFAULT 0.0
                );
            """)
            
            # Knowledge graph analytics table (create extension if not exists)
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            except Exception as e:
                logger.warning(f"⚠️ Could not create vector extension: {str(e)}")
            
            # Use TEXT instead of VECTOR if extension is not available
            try:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_analytics (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        session_id TEXT NOT NULL,
                        query_text TEXT NOT NULL,
                        extracted_concepts JSONB DEFAULT '[]',
                        graph_traversal_path JSONB DEFAULT '[]',
                        semantic_similarity_scores JSONB DEFAULT '{}',
                        voyage_embedding_vector TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            except Exception as e:
                logger.warning(f"⚠️ Using fallback knowledge_analytics schema: {str(e)}")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_analytics (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        session_id TEXT NOT NULL,
                        query_text TEXT NOT NULL,
                        extracted_concepts JSONB DEFAULT '[]',
                        graph_traversal_path JSONB DEFAULT '[]',
                        semantic_similarity_scores JSONB DEFAULT '{}',
                        voyage_embedding_vector TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            
            # Performance indexes
            index_queries = [
                "CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats(user_id);",
                "CREATE INDEX IF NOT EXISTS idx_chats_created_at ON chats(created_at DESC);",
                "CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);",
                "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_messages_search_strategy ON messages(search_strategy);",
                "CREATE INDEX IF NOT EXISTS idx_knowledge_session ON knowledge_analytics(session_id);"
            ]
            
            for query in index_queries:
                try:
                    cursor.execute(query)
                except Exception as e:
                    logger.warning(f"⚠️ Could not create index: {str(e)}")
            
            self.connection.commit()
            logger.info("🛠️ Enhanced GraphRAG database schema created!")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup enhanced tables: {str(e)}")
            if self.connection:
                self.connection.rollback()
    
    def create_chat(self, title: str, user_id: str = "default_user") -> Optional[str]:
        """Create a new chat with enhanced metadata"""
        if not self.connection:
            logger.error("❌ No database connection available")
            return None
            
        try:
            cursor = self.connection.cursor()
            chat_id = str(uuid.uuid4())
            
            cursor.execute("""
                INSERT INTO chats (id, title, user_id)
                VALUES (%s, %s, %s)
                RETURNING id;
            """, (chat_id, title, user_id))
            
            result = cursor.fetchone()
            self.connection.commit()
            
            logger.info(f"✅ Created new enhanced chat: {title}")
            return str(result[0]) if result else chat_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create chat: {str(e)}")
            if self.connection:
                self.connection.rollback()
            return None
    
    def add_enhanced_message(self, chat_id: str, role: str, content: str, 
                           search_metadata: Optional[Dict] = None) -> bool:
        """Add message with comprehensive GraphRAG metadata"""
        if not self.connection:
            logger.error("❌ No database connection available")
            return False
            
        try:
            cursor = self.connection.cursor()
            
            metadata = search_metadata or {}
            
            # Build the insert query dynamically based on available columns
            base_columns = ["chat_id", "role", "content"]
            base_values = [chat_id, role, content]
            
            optional_columns = {
                "context_used": metadata.get('context_used', 0),
                "graph_context_used": metadata.get('graph_context_used', 0),
                "voyage_embedding_used": metadata.get('voyage_embedding_used', False),
                "search_strategy": metadata.get('search_strategy', 'hybrid'),
                "relevance_scores": json.dumps(metadata.get('relevance_scores', {})),
                "extracted_entities": json.dumps(metadata.get('extracted_entities', [])),
                "graph_paths": json.dumps(metadata.get('graph_paths', [])),
                "response_confidence": metadata.get('response_confidence', 0.0),
                "processing_time": metadata.get('processing_time', 0.0),
                "metadata": json.dumps(metadata.get('additional_metadata', {}))
            }
            
            # Check which columns exist and add them to the query
            for col_name, col_value in optional_columns.items():
                if self.column_exists('messages', col_name):
                    base_columns.append(col_name)
                    base_values.append(col_value)
            
            placeholders = ', '.join(['%s'] * len(base_columns))
            columns_str = ', '.join(base_columns)
            
            cursor.execute(f"""
                INSERT INTO messages ({columns_str})
                VALUES ({placeholders});
            """, base_values)
            
            # Update chat statistics if column exists
            if self.column_exists('chats', 'total_messages'):
                cursor.execute("""
                    UPDATE chats SET 
                        updated_at = CURRENT_TIMESTAMP,
                        total_messages = total_messages + 1
                    WHERE id = %s;
                """, (chat_id,))
            else:
                cursor.execute("""
                    UPDATE chats SET 
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s;
                """, (chat_id,))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add enhanced message: {str(e)}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def get_chats(self, user_id: str = "default_user", limit: int = 50) -> List[Dict]:
        """Get chats with enhanced analytics"""
        if not self.connection:
            logger.error("❌ No database connection available")
            return []
            
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Build query based on available columns
            base_query = """
                SELECT c.id, c.title, c.created_at, c.updated_at,
                       COUNT(m.id) as message_count,
                       MAX(m.timestamp) as last_message_time
            """
            
            if self.column_exists('chats', 'user_id'):
                base_query += ", c.user_id"
            if self.column_exists('chats', 'is_active'):
                base_query += ", c.is_active"
            if self.column_exists('chats', 'total_messages'):
                base_query += ", c.total_messages"
            if self.column_exists('messages', 'response_confidence'):
                base_query += ", AVG(m.response_confidence) as avg_confidence"
            if self.column_exists('messages', 'voyage_embedding_used'):
                base_query += ", COUNT(CASE WHEN m.voyage_embedding_used THEN 1 END) as voyage_usage_count"
            
            base_query += """
                FROM chats c
                LEFT JOIN messages m ON c.id = m.chat_id
            """
            
            # Add WHERE clause based on available columns
            where_conditions = []
            params = []
            
            if self.column_exists('chats', 'user_id'):
                where_conditions.append("c.user_id = %s")
                params.append(user_id)
            
            if self.column_exists('chats', 'is_active'):
                where_conditions.append("c.is_active = true")
            
            if where_conditions:
                base_query += " WHERE " + " AND ".join(where_conditions)
            
            base_query += """
                GROUP BY c.id
                ORDER BY COALESCE(MAX(m.timestamp), c.created_at) DESC
                LIMIT %s;
            """
            params.append(limit)
            
            cursor.execute(base_query, params)
            return cursor.fetchall()
            
        except Exception as e:
            logger.error(f"❌ Failed to get enhanced chats: {str(e)}")
            return []
    
    def get_chat_messages(self, chat_id: str) -> List[Dict]:
        """Get messages with available metadata"""
        if not self.connection:
            logger.error("❌ No database connection available")
            return []
            
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM messages 
                WHERE chat_id = %s 
                ORDER BY timestamp ASC;
            """, (chat_id,))
            
            return cursor.fetchall()
            
        except Exception as e:
            logger.error(f"❌ Failed to get messages: {str(e)}")
            return []
    
    def delete_chat(self, chat_id: str) -> bool:
        """Soft delete chat if possible, otherwise hard delete"""
        if not self.connection:
            logger.error("❌ No database connection available")
            return False
            
        try:
            cursor = self.connection.cursor()
            
            if self.column_exists('chats', 'is_active'):
                cursor.execute("UPDATE chats SET is_active = false WHERE id = %s;", (chat_id,))
            else:
                cursor.execute("DELETE FROM chats WHERE id = %s;", (chat_id,))
            
            self.connection.commit()
            logger.info(f"✅ Successfully deleted chat: {chat_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete chat: {str(e)}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def get_chat_analytics(self, chat_id: str) -> Dict:
        """Get comprehensive chat analytics based on available columns"""
        if not self.connection:
            logger.error("❌ No database connection available")
            return {}
            
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Build analytics query based on available columns
            select_parts = ["COUNT(*) as total_messages"]
            
            if self.column_exists('messages', 'response_confidence'):
                select_parts.append("AVG(response_confidence) as avg_confidence")
            if self.column_exists('messages', 'context_used'):
                select_parts.append("AVG(context_used) as avg_context_usage")
            if self.column_exists('messages', 'graph_context_used'):
                select_parts.append("AVG(graph_context_used) as avg_graph_usage")
            if self.column_exists('messages', 'voyage_embedding_used'):
                select_parts.append("COUNT(CASE WHEN voyage_embedding_used THEN 1 END) as voyage_usage")
            if self.column_exists('messages', 'processing_time'):
                select_parts.append("AVG(processing_time) as avg_processing_time")
            if self.column_exists('messages', 'search_strategy'):
                select_parts.append("array_agg(DISTINCT search_strategy) as strategies_used")
            
            query = f"""
                SELECT {', '.join(select_parts)}
                FROM messages 
                WHERE chat_id = %s AND role = 'assistant';
            """
            
            cursor.execute(query, (chat_id,))
            result = cursor.fetchone() or {}
            return dict(result) if result else {}
            
        except Exception as e:
            logger.error(f"❌ Failed to get chat analytics: {str(e)}")
            return {}
    
    def add_knowledge_analytics(self, session_id: str, query_text: str, 
                              extracted_concepts: Optional[List[Dict]] = None,
                              graph_traversal_path: Optional[List[Dict]] = None,
                              semantic_scores: Optional[Dict] = None,
                              voyage_embedding: Optional[List[float]] = None) -> bool:
        """Add knowledge analytics data"""
        if not self.connection:
            logger.error("❌ No database connection available")
            return False
            
        if not self.table_exists('knowledge_analytics'):
            logger.warning("⚠️ Knowledge analytics table doesn't exist")
            return False
            
        try:
            cursor = self.connection.cursor()
            
            # Convert embedding to string if provided
            embedding_str = json.dumps(voyage_embedding) if voyage_embedding else None
            
            cursor.execute("""
                INSERT INTO knowledge_analytics (
                    session_id, query_text, extracted_concepts, 
                    graph_traversal_path, semantic_similarity_scores, 
                    voyage_embedding_vector
                )
                VALUES (%s, %s, %s, %s, %s, %s);
            """, (
                session_id,
                query_text,
                json.dumps(extracted_concepts or []),
                json.dumps(graph_traversal_path or []),
                json.dumps(semantic_scores or {}),
                embedding_str
            ))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add knowledge analytics: {str(e)}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def get_user_stats(self, user_id: str = "default_user") -> Dict:
        """Get comprehensive user statistics based on available columns"""
        if not self.connection:
            logger.error("❌ No database connection available")
            return {}
            
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Build query based on available columns
            select_parts = [
                "COUNT(DISTINCT c.id) as total_chats",
                "COUNT(m.id) as total_messages"
            ]
            
            if self.column_exists('messages', 'response_confidence'):
                select_parts.append("AVG(m.response_confidence) as overall_avg_confidence")
            if self.column_exists('messages', 'processing_time'):
                select_parts.append("AVG(m.processing_time) as overall_avg_processing_time")
            if self.column_exists('messages', 'voyage_embedding_used'):
                select_parts.append("COUNT(CASE WHEN m.voyage_embedding_used THEN 1 END) as total_voyage_usage")
            
            select_parts.extend([
                "MIN(c.created_at) as first_chat_date",
                "MAX(c.updated_at) as last_activity_date"
            ])
            
            query = f"""
                SELECT {', '.join(select_parts)}
                FROM chats c
                LEFT JOIN messages m ON c.id = m.chat_id
            """
            
            params = []
            where_conditions = []
            
            if self.column_exists('chats', 'user_id'):
                where_conditions.append("c.user_id = %s")
                params.append(user_id)
            
            if self.column_exists('chats', 'is_active'):
                where_conditions.append("c.is_active = true")
            
            if where_conditions:
                query += " WHERE " + " AND ".join(where_conditions)
            
            cursor.execute(query, params)
            result = cursor.fetchone() or {}
            return dict(result) if result else {}
            
        except Exception as e:
            logger.error(f"❌ Failed to get user stats: {str(e)}")
            return {}
    
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up old inactive sessions"""
        if not self.connection:
            logger.error("❌ No database connection available")
            return 0
            
        try:
            cursor = self.connection.cursor()
            
            if self.column_exists('chats', 'is_active'):
                cursor.execute("""
                    UPDATE chats SET is_active = false 
                    WHERE updated_at < NOW() - INTERVAL '%s days'
                    AND is_active = true;
                """, (days_old,))
            else:
                cursor.execute("""
                    DELETE FROM chats 
                    WHERE updated_at < NOW() - INTERVAL '%s days';
                """, (days_old,))
            
            deleted_count = cursor.rowcount
            self.connection.commit()
            
            logger.info(f"✅ Cleaned up {deleted_count} old sessions")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old sessions: {str(e)}")
            if self.connection:
                self.connection.rollback()
            return 0
    
    def get_system_health(self) -> Dict:
        """Get system health metrics"""
        if not self.connection:
            logger.error("❌ No database connection available")
            return {"status": "unhealthy", "error": "No database connection"}
            
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Test basic connectivity
            cursor.execute("SELECT 1 as test;")
            cursor.fetchone()
            
            # Get table sizes and counts
            count_queries = []
            if self.table_exists('chats'):
                if self.column_exists('chats', 'is_active'):
                    count_queries.append("(SELECT COUNT(*) FROM chats WHERE is_active = true) as active_chats")
                else:
                    count_queries.append("(SELECT COUNT(*) FROM chats) as active_chats")
            
            if self.table_exists('messages'):
                count_queries.append("(SELECT COUNT(*) FROM messages) as total_messages")
            
            if self.table_exists('knowledge_analytics'):
                count_queries.append("(SELECT COUNT(*) FROM knowledge_analytics) as knowledge_entries")
            
            if count_queries:
                size_queries = [
                    "(SELECT pg_size_pretty(pg_total_relation_size('chats'))) as chats_table_size",
                    "(SELECT pg_size_pretty(pg_total_relation_size('messages'))) as messages_table_size"
                ]
                
                all_queries = count_queries + size_queries
                cursor.execute(f"SELECT {', '.join(all_queries)};")
                health_data = cursor.fetchone()
            else:
                health_data = {}
            
            return {
                "status": "healthy",
                "database_connected": True,
                "metrics": dict(health_data) if health_data else {},
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ System health check failed: {str(e)}")
            return {
                "status": "unhealthy", 
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("🔚 Database connection closed.")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

# Example usage and testing
if __name__ == "__main__":
    # Test the database manager
    db = DatabaseManager()
    
    if db.connection:
        print("Database connection successful!")
        
        # Test creating a chat
        chat_id = db.create_chat("Test Chat", "test_user")
        if chat_id:
            print(f"Created chat: {chat_id}")
            
            # Test adding messages
            db.add_enhanced_message(chat_id, "user", "Hello, this is a test message")
            db.add_enhanced_message(chat_id, "assistant", "Hello! This is a test response.", {
                "search_strategy": "hybrid",
                "response_confidence": 0.95,
                "processing_time": 1.2
            })
            
            # Test getting messages
            messages = db.get_chat_messages(chat_id)
            print(f"Retrieved {len(messages)} messages")
            
            # Test analytics
            analytics = db.get_chat_analytics(chat_id)
            print(f"Analytics: {analytics}")
            
        # Test system health
        health = db.get_system_health()
        print(f"System health: {health}")
        
    else:
        print("Failed to connect to database!")
    
    db.close()