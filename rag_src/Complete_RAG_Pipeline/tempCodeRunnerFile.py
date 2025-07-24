  else:
            answers=await asyncio.gather(*[self._retrieve_and_generate(q,i) for q,i in enumerate(queries)])
            yield "\n\n".join(answers)