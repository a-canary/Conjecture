#!/usr/bin/env python3
"""
Simple GUI implementation using the unified Conjecture API
Demonstrates the clean, direct interface pattern for GUI applications
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from typing import List, Optional

from conjecture import Conjecture

class SimpleGUI:
    """Simple GUI implementation using the unified Conjecture API."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🚀 Conjecture Simple GUI")
        self.root.geometry("800x600")

        # Single unified API - same pattern as CLI and TUI
        self.cf = Conjecture()

        self._setup_ui()

    def _setup_ui(self):
        """Setup the GUI components."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Create tabs
        self._create_search_tab()
        self._create_add_claim_tab()
        self._create_stats_tab()

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _create_search_tab(self):
        """Create the search tab."""
        search_frame = ttk.Frame(self.notebook)
        self.notebook.add(search_frame, text="🔍 Search")

        # Search input
        ttk.Label(search_frame, text="Search Query:").pack(pady=(10, 5))

        search_input_frame = ttk.Frame(search_frame)
        search_input_frame.pack(fill="x", padx=20)

        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_input_frame, textvariable=self.search_var)
        self.search_entry.pack(side="left", fill="x", expand=True)

        ttk.Button(search_input_frame, text="Search", command=self._on_search).pack(
            side="right", padx=(5, 0)
        )

        # Results area
        ttk.Label(search_frame, text="Results:").pack(pady=(20, 5))

        self.results_tree = ttk.Treeview(
            search_frame,
            columns=("id", "content", "confidence", "type"),
            show="headings",
        )
        self.results_tree.heading("id", text="ID")
        self.results_tree.heading("content", text="Content")
        self.results_tree.heading("confidence", text="Confidence")
        self.results_tree.heading("type", text="Type")

        self.results_tree.column("id", width=100)
        self.results_tree.column("content", width=300)
        self.results_tree.column("confidence", width=100)
        self.results_tree.column("type", width=100)

        self.results_tree.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(
            search_frame, orient="vertical", command=self.results_tree.yview
        )
        scrollbar.pack(side="right", fill="y")
        self.results_tree.configure(yscrollcommand=scrollbar.set)

    def _create_add_claim_tab(self):
        """Create the add claim tab."""
        add_frame = ttk.Frame(self.notebook)
        self.notebook.add(add_frame, text="➕ Add Claim")

        # Content input
        ttk.Label(add_frame, text="Claim Content:").pack(pady=(10, 5))

        self.content_text = scrolledtext.ScrolledText(add_frame, height=5, width=70)
        self.content_text.pack(padx=20, fill="x")

        # Confidence and type frame
        input_frame = ttk.Frame(add_frame)
        input_frame.pack(pady=10, padx=20, fill="x")

        # Confidence
        ttk.Label(input_frame, text="Confidence (0.0-1.0):").pack(side="left")
        self.confidence_var = tk.StringVar(value="0.8")
        ttk.Entry(input_frame, textvariable=self.confidence_var, width=10).pack(
            side="left", padx=(5, 20)
        )

        # Claim type
        ttk.Label(input_frame, text="Type:").pack(side="left")
        self.type_var = tk.StringVar(value="concept")
        type_combo = ttk.Combobox(
            input_frame,
            textvariable=self.type_var,
            values=["concept", "reference", "thesis", "example", "goal"],
            state="readonly",
            width=15,
        )
        type_combo.pack(side="left", padx=(5, 0))

        # Tags
        ttk.Label(add_frame, text="Tags (comma-separated):").pack(pady=(10, 5))
        self.tags_var = tk.StringVar()
        ttk.Entry(add_frame, textvariable=self.tags_var, width=70).pack(
            padx=20, fill="x"
        )

        # Add button
        ttk.Button(add_frame, text="Add Claim", command=self._on_add_claim).pack(
            pady=20
        )

    def _create_stats_tab(self):
        """Create the statistics tab."""
        stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_frame, text="📊 Statistics")

        # Stats display
        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=20, width=80)
        self.stats_text.pack(padx=20, pady=20, fill="both", expand=True)

        # Refresh button
        ttk.Button(
            stats_frame, text="Refresh Statistics", command=self._on_refresh_stats
        ).pack(pady=(0, 10))

    def _on_search(self):
        """Handle search button click."""
        query = self.search_var.get().strip()
        if not query:
            messagebox.showwarning("Warning", "Please enter a search query")
            return

        if len(query) < 5:
            messagebox.showwarning(
                "Warning", "Query must be at least 5 characters long"
            )
            return

        try:
            self.status_var.set("Searching...")
            self.root.update()

            # Use unified API directly
            result = self.cf.explore(query, max_claims=20)

            # Clear existing results
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)

            # Add new results
            for claim in result.claims:
                type_str = ", ".join([t.value for t in claim.type])
                content = (
                    claim.content[:50] + "..."
                    if len(claim.content) > 50
                    else claim.content
                )

                self.results_tree.insert(
                    "",
                    "end",
                    values=(claim.id, content, f"{claim.confidence:.2f}", type_str),
                )

            self.status_var.set(
                f"Found {len(result.claims)} claims in {result.search_time:.2f}s"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Search failed: {str(e)}")
            self.status_var.set("Search failed")

    def _on_add_claim(self):
        """Handle add claim button click."""
        content = self.content_text.get("1.0", tk.END).strip()
        confidence_str = self.confidence_var.get()
        claim_type = self.type_var.get()
        tags_str = self.tags_var.get().strip()

        # Validation
        if not content or len(content) < 10:
            messagebox.showwarning(
                "Warning", "Content must be at least 10 characters long"
            )
            return

        try:
            confidence = float(confidence_str)
            if not (0.0 <= confidence <= 1.0):
                raise ValueError()
        except ValueError:
            messagebox.showwarning(
                "Warning", "Confidence must be a number between 0.0 and 1.0"
            )
            return

        tags = (
            [tag.strip() for tag in tags_str.split(",") if tag.strip()]
            if tags_str
            else []
        )

        try:
            self.status_var.set("Adding claim...")
            self.root.update()

            # Use unified API directly
            claim = self.cf.add_claim(content, confidence, claim_type, tags)

            messagebox.showinfo(
                "Success",
                f"Claim created successfully!\n\nID: {claim.id}\nType: {claim_type}",
            )

            # Clear form
            self.content_text.delete("1.0", tk.END)
            self.confidence_var.set("0.8")
            self.type_var.set("concept")
            self.tags_var.set("")

            self.status_var.set(f"Claim {claim.id} created successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to add claim: {str(e)}")
            self.status_var.set("Failed to add claim")

    def _on_refresh_stats(self):
        """Handle refresh statistics button click."""
        try:
            self.status_var.set("Loading statistics...")
            self.root.update()

            # Use unified API directly
            stats = self.cf.get_statistics()

            # Display stats
            self.stats_text.delete("1.0", tk.END)
            self.stats_text.insert(tk.END, "📊 Conjecture System Statistics\n")
            self.stats_text.insert(tk.END, "=" * 40 + "\n\n")

            for key, value in stats.items():
                display_key = key.replace("_", " ").title()
                self.stats_text.insert(tk.END, f"{display_key}: {value}\n")

            self.status_var.set("Statistics updated")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load statistics: {str(e)}")
            self.status_var.set("Failed to load statistics")

    def run(self):
        """Run the GUI application."""
        self.root.mainloop()

def main():
    """Run the simple GUI."""
    try:
        gui = SimpleGUI()
        gui.run()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start GUI: {str(e)}")

if __name__ == "__main__":
    main()
