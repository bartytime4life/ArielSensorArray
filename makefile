.PHONY: push pushm push-amend

# Reuse the last commit message automatically.
push:
	@branch=$$(git rev-parse --abbrev-ref HEAD); \
	msg=$$(git log -1 --pretty=%B 2>/dev/null || echo "chore(update): sync changes"); \
	git add -A; \
	if git diff --cached --quiet; then echo "No changes to commit."; exit 0; fi; \
	git commit -m "$$msg" || true; \
	git push origin "$$branch"

# Push with a custom message: make pushm M="feat: add training loop"
pushm:
	@branch=$$(git rev-parse --abbrev-ref HEAD); \
	test -n "$$M" || { echo "Usage: make pushm M='your message'"; exit 2; }; \
	git add -A; \
	git commit -m "$$M" || true; \
	git push origin "$$branch"

# Amend the previous commit (no new message) and push safely.
push-amend:
	@branch=$$(git rev-parse --abbrev-ref HEAD); \
	git add -A; \
	git commit --amend --no-edit || true; \
	git push --force-with-lease origin "$$branch"
