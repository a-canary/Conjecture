async def generate_claims_with_context(self, query: str):
    """Generate initial claims using LLM with context awareness"""
    try:
            # OPTIMIZATION: Parallel context collection
            context_start = time.time()
            
            # Get relevant context for query with caching and XML optimization
            context_claims = await self._collect_context_cached(
                query, {"task": "exploration"}, max_skills=3, max_samples=5
            )
            
            context_time = time.time() - context_start
            self._performance_stats["context_collection_time"].append(context_time)

            # Build context string using enhanced template manager
            context_claims = await self.enhanced_template_manager.get_context_for_task(
                query, {"task": "exploration"}, max_skills=3, max_samples=5
            )
            
            # Build context string with XML optimization
            context_string = self.enhanced_template_manager.build_context_string(context_claims)
            
            # Get enhanced XML template for claim creation
            enhanced_template = self.enhanced_template_manager.get_template("research_enhanced_xml")
            
            prompt = enhanced_template.format(
                user_query=query,
                relevant_context=context_string
            )
            
            llm_request = LLMRequest(
                prompt=prompt, max_tokens=3000, temperature=0.7, task_type="explore"
            )

            response = self.llm_bridge.process(llm_request)

            if response.success:
                return self._parse_claims_from_response(response.content)
            else:
                raise Exception(f"LLM processing failed: {response.errors}")
    except Exception as e:
        raise Exception(f"Claim generation failed: {str(e)}")