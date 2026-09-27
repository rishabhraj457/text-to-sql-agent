import os
import sys
import argparse
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from rich.console import Console
from rich.panel import Panel
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

console = Console()

def main():
    """Main entry point for the SQL Agent CLI"""
    parser = argparse.ArgumentParser(
        description="Text-to-SQL Agent powered by LangChain and open-source models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py "What are the top 5 best-selling artists?"
  python agent.py "Which employee generated the most revenue?"
  python agent.py "How many customers are from Canada?"
        """
    )
    parser.add_argument(
        "question",
        type=str,
        help="Natural language question to answer using the Chinook database"
    )

    args = parser.parse_args()

    # Display the question
    console.print(Panel(
        f"[bold cyan]Question:[/bold cyan] {args.question}",
        border_style="cyan"
    ))
    console.print()

    # Connect to Chinook database
    console.print("[dim]Connecting to database...[/dim]")
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chinook.db")
    db = SQLDatabase.from_uri(
        f"sqlite:///{db_path}",
        sample_rows_in_table_info=3
    )

    # Initialize Llama 3 on Groq
    model = ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0
    )

    console.print("[dim]Generating SQL query...[/dim]")
    
    try:
        # Get database schema
        schema = db.get_table_info()
        
        # 1. Generate the SQL query
        sql_prompt = f"""You are a SQLite expert. Given the database schema below, write a syntactically correct SQLite query to answer the user's question.
Unless the user specifies a number of examples, limit your query to at most 5 results using LIMIT.
Return ONLY the SQL query, without any markdown formatting or explanation.

Schema:
{schema}

Question: {args.question}
SQL Query:"""

        sql_response = model.invoke(sql_prompt)
        sql_query = sql_response.content.strip()
        
        # Clean markdown formatting if present
        if "```sql" in sql_query:
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].strip()
            
        console.print(f"[dim]Executing:[/dim] [cyan]{sql_query}[/cyan]\n")
        
        # Execute the SQL query
        result = db.run(sql_query)
        
        # 3. Formulate the final answer using the model
        final_prompt = f"Given the following user question, corresponding SQL query, and SQL result, answer the user question directly and concisely.\n\nQuestion: {args.question}\nSQL Query: {sql_query}\nSQL Result: {result}\n\nAnswer:"
        
        answer_msg = model.invoke(final_prompt)
        answer = answer_msg.content if hasattr(answer_msg, 'content') else str(answer_msg)
        
        # Display the result
        console.print(Panel(
            f"[bold green]Answer:[/bold green]\n\n{answer}",
            border_style="green"
        ))

    except Exception as e:
        console.print(Panel(
            f"[bold red]Error:[/bold red]\n\n{str(e)}",
            border_style="red"
        ))
        sys.exit(1)


if __name__ == "__main__":
    main()
