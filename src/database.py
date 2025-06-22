import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import pickle

class ResearchDatabase:
    """Database for storing and tracking quantum optics research evolution"""
    
    def __init__(self, db_path: str = "data/research_history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.logger = logging.getLogger('ResearchDatabase')
        
        # Initialize database
        self._create_tables()
        
    def _create_tables(self) -> None:
        """Create database tables for research tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- Research solutions table
                CREATE TABLE IF NOT EXISTS solutions (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    generation INTEGER NOT NULL,
                    parent_ids TEXT,  -- JSON array of parent IDs
                    mutation_type TEXT,
                    total_score REAL NOT NULL,
                    feasibility_score REAL,
                    mathematics_score REAL,
                    novelty_score REAL,
                    performance_score REAL,
                    evaluation_details TEXT,  -- JSON
                    generation_metadata TEXT,  -- JSON
                    timestamp REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Evolution statistics table
                CREATE TABLE IF NOT EXISTS evolution_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    generation INTEGER NOT NULL,
                    population_size INTEGER NOT NULL,
                    best_score REAL NOT NULL,
                    average_score REAL NOT NULL,
                    diversity_index REAL NOT NULL,
                    new_solutions INTEGER NOT NULL,
                    breakthrough_solutions INTEGER NOT NULL,
                    convergence_rate REAL NOT NULL,
                    model_used TEXT,
                    timestamp REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Breakthrough discoveries table
                CREATE TABLE IF NOT EXISTS breakthroughs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    solution_id TEXT NOT NULL,
                    discovery_type TEXT,
                    significance_score REAL,
                    novelty_indicators TEXT,  -- JSON array
                    potential_applications TEXT,  -- JSON array
                    experimental_feasibility REAL,
                    theoretical_soundness REAL,
                    discovery_timestamp REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (solution_id) REFERENCES solutions (id)
                );
                
                -- Model performance tracking
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    solutions_generated INTEGER NOT NULL,
                    average_score REAL NOT NULL,
                    best_score REAL NOT NULL,
                    total_tokens INTEGER,
                    total_cost REAL,
                    generation_time REAL,
                    physics_accuracy REAL,
                    creativity_score REAL,
                    timestamp REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Research lineage for tracking evolution paths
                CREATE TABLE IF NOT EXISTS lineage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    child_id TEXT NOT NULL,
                    parent_id TEXT NOT NULL,
                    mutation_type TEXT,
                    generation_gap INTEGER,
                    score_improvement REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (child_id) REFERENCES solutions (id),
                    FOREIGN KEY (parent_id) REFERENCES solutions (id)
                );
                
                -- Create indexes for better query performance
                CREATE INDEX IF NOT EXISTS idx_solutions_generation ON solutions (generation);
                CREATE INDEX IF NOT EXISTS idx_solutions_score ON solutions (total_score);
                CREATE INDEX IF NOT EXISTS idx_solutions_category ON solutions (category);
                CREATE INDEX IF NOT EXISTS idx_solutions_timestamp ON solutions (timestamp);
                CREATE INDEX IF NOT EXISTS idx_evolution_generation ON evolution_stats (generation);
                CREATE INDEX IF NOT EXISTS idx_breakthroughs_score ON breakthroughs (significance_score);
                CREATE INDEX IF NOT EXISTS idx_model_perf_model ON model_performance (model_name);
                CREATE INDEX IF NOT EXISTS idx_lineage_child ON lineage (child_id);
                CREATE INDEX IF NOT EXISTS idx_lineage_parent ON lineage (parent_id);
            """)
            
    def store_solution(self, solution_data: Dict[str, Any]) -> bool:
        """Store a research solution in the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Extract evaluation results
                eval_result = solution_data.get('evaluation_result', {})
                
                conn.execute("""
                    INSERT OR REPLACE INTO solutions (
                        id, content, category, title, description, generation,
                        parent_ids, mutation_type, total_score, feasibility_score,
                        mathematics_score, novelty_score, performance_score,
                        evaluation_details, generation_metadata, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    solution_data['id'],
                    solution_data['content'],
                    solution_data['category'],
                    solution_data['title'],
                    solution_data.get('description', ''),
                    solution_data['generation'],
                    json.dumps(solution_data.get('parent_ids', [])),
                    solution_data.get('mutation_type', ''),
                    eval_result.get('total_score', 0.0),
                    eval_result.get('feasibility', 0.0),
                    eval_result.get('mathematics', 0.0),
                    eval_result.get('novelty', 0.0),
                    eval_result.get('performance', 0.0),
                    json.dumps(eval_result.get('details', {})),
                    json.dumps(solution_data.get('generation_metadata', {})),
                    solution_data.get('timestamp', datetime.now().timestamp())
                ))
                
                # Store lineage information
                self._store_lineage(solution_data)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store solution {solution_data.get('id', 'unknown')}: {e}")
            return False
            
    def _store_lineage(self, solution_data: Dict[str, Any]) -> None:
        """Store lineage information for evolution tracking"""
        parent_ids = solution_data.get('parent_ids', [])
        if not parent_ids:
            return
            
        with sqlite3.connect(self.db_path) as conn:
            for parent_id in parent_ids:
                if parent_id:  # Skip empty parent IDs
                    # Get parent generation for gap calculation
                    parent_gen = conn.execute(
                        "SELECT generation, total_score FROM solutions WHERE id = ?", 
                        (parent_id,)
                    ).fetchone()
                    
                    if parent_gen:
                        generation_gap = solution_data['generation'] - parent_gen[0]
                        parent_score = parent_gen[1]
                        current_score = solution_data.get('evaluation_result', {}).get('total_score', 0)
                        score_improvement = current_score - parent_score
                        
                        conn.execute("""
                            INSERT INTO lineage (
                                child_id, parent_id, mutation_type, 
                                generation_gap, score_improvement
                            ) VALUES (?, ?, ?, ?, ?)
                        """, (
                            solution_data['id'],
                            parent_id,
                            solution_data.get('mutation_type', ''),
                            generation_gap,
                            score_improvement
                        ))
                        
    def store_evolution_stats(self, stats_data: Dict[str, Any]) -> bool:
        """Store evolution statistics for a generation"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO evolution_stats (
                        generation, population_size, best_score, average_score,
                        diversity_index, new_solutions, breakthrough_solutions,
                        convergence_rate, model_used, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    stats_data['generation'],
                    stats_data['population_size'],
                    stats_data['best_score'],
                    stats_data['average_score'],
                    stats_data['diversity_index'],
                    stats_data['new_solutions'],
                    stats_data['breakthrough_solutions'],
                    stats_data['convergence_rate'],
                    stats_data.get('model_used', ''),
                    stats_data.get('timestamp', datetime.now().timestamp())
                ))
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store evolution stats: {e}")
            return False
            
    def store_breakthrough(self, solution_id: str, breakthrough_data: Dict[str, Any]) -> bool:
        """Store breakthrough discovery information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO breakthroughs (
                        solution_id, discovery_type, significance_score,
                        novelty_indicators, potential_applications,
                        experimental_feasibility, theoretical_soundness,
                        discovery_timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    solution_id,
                    breakthrough_data.get('discovery_type', ''),
                    breakthrough_data.get('significance_score', 0.0),
                    json.dumps(breakthrough_data.get('novelty_indicators', [])),
                    json.dumps(breakthrough_data.get('potential_applications', [])),
                    breakthrough_data.get('experimental_feasibility', 0.0),
                    breakthrough_data.get('theoretical_soundness', 0.0),
                    breakthrough_data.get('timestamp', datetime.now().timestamp())
                ))
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store breakthrough for {solution_id}: {e}")
            return False
            
    def store_model_performance(self, performance_data: Dict[str, Any]) -> bool:
        """Store model performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO model_performance (
                        model_name, generation, solutions_generated,
                        average_score, best_score, total_tokens, total_cost,
                        generation_time, physics_accuracy, creativity_score, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    performance_data['model_name'],
                    performance_data['generation'],
                    performance_data['solutions_generated'],
                    performance_data['average_score'],
                    performance_data['best_score'],
                    performance_data.get('total_tokens', 0),
                    performance_data.get('total_cost', 0.0),
                    performance_data.get('generation_time', 0.0),
                    performance_data.get('physics_accuracy', 0.0),
                    performance_data.get('creativity_score', 0.0),
                    performance_data.get('timestamp', datetime.now().timestamp())
                ))
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store model performance: {e}")
            return False
            
    def get_best_solutions(self, limit: int = 10, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve best solutions from database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = """
                SELECT * FROM solutions 
                WHERE 1=1
            """
            params = []
            
            if category:
                query += " AND category = ?"
                params.append(category)
                
            query += " ORDER BY total_score DESC LIMIT ?"
            params.append(limit)
            
            results = conn.execute(query, params).fetchall()
            
        return [dict(row) for row in results]
        
    def get_evolution_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get evolution statistics history"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM evolution_stats ORDER BY generation"
            if limit:
                query += f" LIMIT {limit}"
                
            results = conn.execute(query).fetchall()
            
        return [dict(row) for row in results]
        
    def get_breakthroughs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get breakthrough discoveries"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            results = conn.execute("""
                SELECT b.*, s.title, s.category, s.total_score
                FROM breakthroughs b
                JOIN solutions s ON b.solution_id = s.id
                ORDER BY b.significance_score DESC
                LIMIT ?
            """, (limit,)).fetchall()
            
        return [dict(row) for row in results]
        
    def get_solution_lineage(self, solution_id: str) -> List[Dict[str, Any]]:
        """Get the evolutionary lineage of a solution"""
        lineage = []
        current_id = solution_id
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            while current_id:
                # Get current solution
                solution = conn.execute(
                    "SELECT * FROM solutions WHERE id = ?", 
                    (current_id,)
                ).fetchone()
                
                if solution:
                    lineage.append(dict(solution))
                    
                    # Find parent
                    parent_ids = json.loads(solution['parent_ids'])
                    current_id = parent_ids[0] if parent_ids else None
                else:
                    break
                    
        return lineage
        
    def get_model_comparison(self) -> Dict[str, Any]:
        """Get comparative model performance statistics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            results = conn.execute("""
                SELECT 
                    model_name,
                    COUNT(*) as total_generations,
                    AVG(average_score) as avg_performance,
                    MAX(best_score) as best_performance,
                    SUM(total_tokens) as total_tokens_used,
                    SUM(total_cost) as total_cost,
                    AVG(generation_time) as avg_generation_time,
                    AVG(physics_accuracy) as avg_physics_accuracy,
                    AVG(creativity_score) as avg_creativity
                FROM model_performance
                GROUP BY model_name
                ORDER BY avg_performance DESC
            """).fetchall()
            
        return [dict(row) for row in results]
        
    def get_category_statistics(self) -> Dict[str, Any]:
        """Get statistics by research category"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            results = conn.execute("""
                SELECT 
                    category,
                    COUNT(*) as total_solutions,
                    AVG(total_score) as average_score,
                    MAX(total_score) as best_score,
                    AVG(feasibility_score) as avg_feasibility,
                    AVG(mathematics_score) as avg_mathematics,
                    AVG(novelty_score) as avg_novelty,
                    AVG(performance_score) as avg_performance
                FROM solutions
                GROUP BY category
                ORDER BY average_score DESC
            """).fetchall()
            
        return [dict(row) for row in results]
        
    def search_solutions(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search solutions by content"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            results = conn.execute("""
                SELECT * FROM solutions 
                WHERE content LIKE ? OR title LIKE ? OR description LIKE ?
                ORDER BY total_score DESC
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", f"%{query}%", limit)).fetchall()
            
        return [dict(row) for row in results]
        
    def get_convergence_analysis(self) -> Dict[str, Any]:
        """Analyze convergence patterns in evolution"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get convergence data
            convergence_data = conn.execute("""
                SELECT generation, best_score, average_score, diversity_index
                FROM evolution_stats
                ORDER BY generation
            """).fetchall()
            
            # Get mutation success rates
            mutation_success = conn.execute("""
                SELECT 
                    mutation_type,
                    COUNT(*) as attempts,
                    AVG(total_score) as avg_score,
                    COUNT(CASE WHEN total_score > 0.8 THEN 1 END) as high_scores
                FROM solutions
                WHERE mutation_type IS NOT NULL
                GROUP BY mutation_type
            """).fetchall()
            
        return {
            'convergence_data': [dict(row) for row in convergence_data],
            'mutation_success_rates': [dict(row) for row in mutation_success]
        }
        
    def export_research_data(self, filepath: str, format: str = 'json') -> bool:
        """Export research data to file"""
        try:
            data = {
                'solutions': self.get_best_solutions(limit=1000),
                'evolution_history': self.get_evolution_history(),
                'breakthroughs': self.get_breakthroughs(limit=100),
                'model_comparison': self.get_model_comparison(),
                'category_statistics': self.get_category_statistics(),
                'convergence_analysis': self.get_convergence_analysis()
            }
            
            if format.lower() == 'json':
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            elif format.lower() == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export data: {e}")
            return False
            
    def cleanup_old_data(self, days_old: int = 30) -> int:
        """Clean up old data beyond specified days"""
        cutoff_timestamp = datetime.now().timestamp() - (days_old * 24 * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            # Clean up old solutions (keep high-scoring ones)
            result = conn.execute("""
                DELETE FROM solutions 
                WHERE timestamp < ? AND total_score < 0.5
            """, (cutoff_timestamp,))
            
            cleaned_count = result.rowcount
            
            # Clean up old evolution stats
            conn.execute("""
                DELETE FROM evolution_stats 
                WHERE timestamp < ?
            """, (cutoff_timestamp,))
            
            # Clean up orphaned lineage records
            conn.execute("""
                DELETE FROM lineage 
                WHERE child_id NOT IN (SELECT id FROM solutions)
                   OR parent_id NOT IN (SELECT id FROM solutions)
            """)
            
        self.logger.info(f"Cleaned up {cleaned_count} old records")
        return cleaned_count
        
    def get_database_stats(self) -> Dict[str, int]:
        """Get database size statistics"""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            tables = ['solutions', 'evolution_stats', 'breakthroughs', 'model_performance', 'lineage']
            for table in tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats[table] = count
                
        return stats