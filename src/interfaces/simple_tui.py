#!/usr/bin/env python3
"""
Simple TUI implementation using the unified Conjecture API
Demonstrates the clean, direct interface pattern for terminal UI
"""

import curses
from typing import List, Optional

from contextflow import Conjecture


class SimpleTUI:
    """Simple TUI implementation using the unified Conjecture API."""

    def __init__(self):
        # Single unified API - same pattern as CLI and GUI
        self.cf = Conjecture()
        self.current_screen = "menu"

    def run(self):
        """Run the TUI application."""
        curses.wrapper(self._main_loop)

    def _main_loop(self, stdscr):
        """Main TUI loop."""
        stdscr.clear()
        curses.curs_set(0)
        stdscr.addstr(0, 0, "🚀 Conjecture Simple TUI", curses.A_BOLD)
        stdscr.addstr(1, 0, "=" * 40)
        
        while True:
            if self.current_screen == "menu":
                self._show_menu(stdscr)
            elif self.current_screen == "search":
                self._show_search(stdscr)
            elif self.current_screen == "add":
                self._show_add_claim(stdscr)
            elif self.current_screen == "stats":
                self._show_stats(stdscr)

    def _show_menu(self, stdscr):
        """Show main menu."""
        stdscr.clear()
        stdscr.addstr(0, 0, "🚀 Conjecture Simple TUI", curses.A_BOLD)
        stdscr.addstr(1, 0, "=" * 40)
        stdscr.addstr(3, 0, "Choose an option:")
        stdscr.addstr(5, 0, "1. Search claims")
        stdscr.addstr(6, 0, "2. Add claim")
        stdscr.addstr(7, 0, "3. View statistics")
        stdscr.addstr(8, 0, "4. Exit")
        stdscr.addstr(10, 0, "Enter choice (1-4): ")
        
        stdscr.refresh()
        
        try:
            choice = stdscr.getch()
            if choice == ord('1'):
                self.current_screen = "search"
            elif choice == ord('2'):
                self.current_screen = "add"
            elif choice == ord('3'):
                self.current_screen = "stats"
            elif choice == ord('4'):
                exit()
        except:
            pass

    def _show_search(self, stdscr):
        """Show search screen."""
        stdscr.clear()
        stdscr.addstr(0, 0, "🔍 Search Claims", curses.A_BOLD)
        stdscr.addstr(1, 0, "=" * 40)
        stdscr.addstr(3, 0, "Enter search query (or 'back' to return): ")
        
        stdscr.refresh()
        curses.echo()
        curses.curs_set(1)
        
        try:
            query = stdscr.getstr(4, 0).decode('utf-8')
            curses.noecho()
            curses.curs_set(0)
            
            if query.lower() == 'back':
                self.current_screen = "menu"
                return
            
            if query.strip():
                # Use unified API directly
                result = self.cf.explore(query, max_claims=5)
                self._show_search_results(stdscr, result)
            else:
                stdscr.addstr(6, 0, "Please enter a search query")
                stdscr.getch()
                
        except:
            curses.noecho()
            curses.curs_set(0)

    def _show_search_results(self, stdscr, result):
        """Show search results."""
        stdscr.clear()
        stdscr.addstr(0, 0, f"🔍 Results for: '{result.query}'", curses.A_BOLD)
        stdscr.addstr(1, 0, "=" * 50)
        stdscr.addstr(2, 0, f"Found {len(result.claims)} claims in {result.search_time:.2f}s")
        stdscr.addstr(3, 0, "")
        
        y_pos = 4
        for i, claim in enumerate(result.claims, 1):
            type_str = ", ".join([t.value for t in claim.type])
            content = claim.content[:50] + "..." if len(claim.content) > 50 else claim.content
            
            stdscr.addstr(y_pos, 0, f"{i}. [{claim.id}]")
            stdscr.addstr(y_pos + 1, 2, f"Content: {content}")
            stdscr.addstr(y_pos + 2, 2, f"Confidence: {claim.confidence:.2f} | Type: {type_str}")
            y_pos += 4
            
            if y_pos > 20:  # Prevent overflow
                break
        
        stdscr.addstr(y_pos + 1, 0, "")
        stdscr.addstr(y_pos + 2, 0, "Press any key to continue...")
        stdscr.refresh()
        stdscr.getch()
        self.current_screen = "menu"

    def _show_add_claim(self, stdscr):
        """Show add claim screen."""
        stdscr.clear()
        stdscr.addstr(0, 0, "➕ Add New Claim", curses.A_BOLD)
        stdscr.addstr(1, 0, "=" * 40)
        
        stdscr.addstr(3, 0, "Enter claim content:")
        stdscr.refresh()
        curses.echo()
        curses.curs_set(1)
        
        try:
            content = stdscr.getstr(4, 0).decode('utf-8')
            
            if not content.strip() or len(content.strip()) < 10:
                stdscr.addstr(6, 0, "Content must be at least 10 characters")
                stdscr.getch()
                curses.noecho()
                curses.curs_set(0)
                self.current_screen = "menu"
                return
            
            stdscr.addstr(6, 0, "Enter confidence (0.0-1.0): ")
            confidence_str = stdscr.getstr(6, 26).decode('utf-8')
            
            try:
                confidence = float(confidence_str)
                if not (0.0 <= confidence <= 1.0):
                    raise ValueError()
            except:
                stdscr.addstr(8, 0, "Invalid confidence. Using 0.8")
                confidence = 0.8
            
            stdscr.addstr(8, 0, "Enter claim type (concept/reference/thesis/example/goal): ")
            claim_type = stdscr.getstr(8, 55).decode('utf-8').lower()
            
            if claim_type not in ['concept', 'reference', 'thesis', 'example', 'goal']:
                stdscr.addstr(10, 0, "Invalid type. Using 'concept'")
                claim_type = 'concept'
            
            curses.noecho()
            curses.curs_set(0)
            
            # Use unified API directly
            claim = self.cf.add_claim(content, confidence, claim_type)
            
            stdscr.clear()
            stdscr.addstr(0, 0, "✅ Claim Created Successfully!", curses.A_BOLD)
            stdscr.addstr(1, 0, "=" * 40)
            stdscr.addstr(3, 0, f"ID: {claim.id}")
            stdscr.addstr(4, 0, f"Content: {claim.content}")
            stdscr.addstr(5, 0, f"Confidence: {claim.confidence:.2f}")
            stdscr.addstr(6, 0, f"Type: {', '.join([t.value for t in claim.type])}")
            stdscr.addstr(8, 0, "Press any key to continue...")
            stdscr.refresh()
            stdscr.getch()
            
        except Exception as e:
            curses.noecho()
            curses.curs_set(0)
            stdscr.addstr(10, 0, f"Error: {str(e)}")
            stdscr.getch()
        
        self.current_screen = "menu"

    def _show_stats(self, stdscr):
        """Show statistics screen."""
        stdscr.clear()
        stdscr.addstr(0, 0, "📊 System Statistics", curses.A_BOLD)
        stdscr.addstr(1, 0, "=" * 40)
        
        try:
            # Use unified API directly
            stats = self.cf.get_statistics()
            
            y_pos = 3
            for key, value in stats.items():
                display_key = key.replace("_", " ").title()
                stdscr.addstr(y_pos, 0, f"{display_key}: {value}")
                y_pos += 1
            
        except Exception as e:
            stdscr.addstr(3, 0, f"Error retrieving statistics: {str(e)}")
        
        stdscr.addstr(y_pos + 2, 0, "Press any key to continue...")
        stdscr.refresh()
        stdscr.getch()
        self.current_screen = "menu"


def main():
    """Run the simple TUI."""
    try:
        tui = SimpleTUI()
        tui.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()