.PHONY: version bump-major bump-minor bump-patch bump-set release

# Show current version
version:
	@cat VERSION

# Bump major version (X+1.0.0)
bump-major:
	@v=$$(bin/version_tools.py bump-major); \
	git add VERSION pyproject.toml src/asa/__init__.py; \
	git commit -m "chore(version): bump to $$v" || true; \
	git tag -a v$$v -m "Release v$$v"; \
	git push origin main; \
	git push origin v$$v; \
	echo "✅ Released v$$v"

# Bump minor version (X.Y+1.0)
bump-minor:
	@v=$$(bin/version_tools.py bump-minor); \
	git add VERSION pyproject.toml src/asa/__init__.py; \
	git commit -m "chore(version): bump to $$v" || true; \
	git tag -a v$$v -m "Release v$$v"; \
	git push origin main; \
	git push origin v$$v; \
	echo "✅ Released v$$v"

# Bump patch version (X.Y.Z+1)
bump-patch:
	@v=$$(bin/version_tools.py bump-patch); \
	git add VERSION pyproject.toml src/asa/__init__.py; \
	git commit -m "chore(version): bump to $$v" || true; \
	git tag -a v$$v -m "Release v$$v"; \
	git push origin main; \
	git push origin v$$v; \
	echo "✅ Released v$$v"

# Set an exact version (use: make bump-set V=0.3.0)
bump-set:
	@test -n "$(V)" || (echo "Usage: make bump-set V=0.3.0" && exit 2) ; \
	v=$$(bin/version_tools.py set "$(V)"); \
	git add VERSION pyproject.toml src/asa/__init__.py; \
	git commit -m "chore(version): set to $$v" || true; \
	git tag -a v$$v -m "Release v$$v"; \
	git push origin main; \
	git push origin v$$v; \
	echo "✅ Released v$$v"

# Tag current VERSION without bumping
release:
	@v=$$(cat VERSION); \
	git tag -a v$$v -m "Release v$$v" || true; \
	git push origin v$$v
