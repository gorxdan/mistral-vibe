from __future__ import annotations

from vibe.core.tools.command_safety import (
    allowlisted_argument_is_unsafe,
    destructive_command_reason,
)


class TestDestructiveDetection:
    def test_rm_recursive_flagged(self):
        assert destructive_command_reason(["rm -rf build/"]) is not None
        assert destructive_command_reason(["rm -r build/"]) is not None
        reason = destructive_command_reason(["rm -rf /"])
        assert reason is not None
        assert "destructive" in reason

    def test_rm_force_flagged(self):
        assert destructive_command_reason(["rm -f foo"]) is not None

    def test_rm_combined_short_flags_flagged(self):
        assert destructive_command_reason(["rm -fr build"]) is not None
        assert destructive_command_reason(["rm -Rf build"]) is not None

    def test_rm_long_flags_flagged(self):
        assert destructive_command_reason(["rm --recursive build"]) is not None
        assert destructive_command_reason(["rm --force foo"]) is not None

    def test_bare_rm_not_flagged(self):
        # A single-file rm without force/recursive is left to the normal flow.
        assert destructive_command_reason(["rm foo.txt"]) is None

    def test_rm_separator_not_treated_as_flag(self):
        # `rm -- -rf` is a file named -rf, not the recursive flag.
        assert destructive_command_reason(["rm -- -rf"]) is None

    def test_sudo_unwrapped(self):
        reason = destructive_command_reason(["sudo rm -rf /"])
        assert reason is not None
        assert "destructive" in reason

    def test_env_wrapper_with_assignments_unwrapped(self):
        reason = destructive_command_reason(["env FOO=bar rm -rf build"])
        assert reason is not None

    def test_nohup_wrapper_unwrapped(self):
        assert destructive_command_reason(["nohup rm -rf build"]) is not None

    def test_chmod_open_mode_flagged(self):
        assert destructive_command_reason(["chmod 777 ."]) is not None
        assert destructive_command_reason(["chmod -R 777 src"]) is not None

    def test_chmod_normal_mode_not_flagged(self):
        assert destructive_command_reason(["chmod +x deploy.sh"]) is None
        assert destructive_command_reason(["chmod 644 file"]) is None

    def test_dd_to_block_device_flagged(self):
        assert destructive_command_reason(["dd if=img.iso of=/dev/sda"]) is not None

    def test_dd_to_dev_null_not_flagged(self):
        assert destructive_command_reason(["echo x | dd of=/dev/null"]) is None

    def test_mkfs_flagged(self):
        assert destructive_command_reason(["mkfs.ext4 /dev/sda1"]) is not None

    def test_multiple_commands_first_reason_wins(self):
        reason = destructive_command_reason(["echo hi", "rm -rf build"])
        assert reason is not None
        assert "build" in reason

    def test_safe_commands_not_flagged(self):
        for parts in (["ls -la"], ["git status"], ["echo hello"], ["grep foo file"]):
            assert destructive_command_reason(parts) is None


class TestArgumentGate:
    def test_find_delete_unsafe(self):
        reason = allowlisted_argument_is_unsafe("find . -delete")
        assert reason is not None
        assert "find" in reason

    def test_find_fls_unsafe(self):
        for pred in ("-fls", "-fprint", "-fprint0", "-fprintf"):
            assert allowlisted_argument_is_unsafe(f"find . {pred} out") is not None

    def test_find_plain_safe(self):
        assert allowlisted_argument_is_unsafe("find . -name foo") is None

    def test_find_exec_handled_elsewhere_not_here(self):
        # The -exec family is routed by the bash guardrail path, not this gate.
        assert allowlisted_argument_is_unsafe("find . -exec id \\;") is None

    def test_non_find_safe(self):
        assert allowlisted_argument_is_unsafe("git status") is None
        assert allowlisted_argument_is_unsafe("ls -la") is None

    def test_ruff_requires_an_explicit_read_only_mode(self):
        assert allowlisted_argument_is_unsafe("ruff check --no-fix .") is None
        assert allowlisted_argument_is_unsafe("ruff format --check .") is None
        assert allowlisted_argument_is_unsafe("ruff format --diff .") is None
        assert allowlisted_argument_is_unsafe("ruff check .") is not None

    def test_ruff_global_options_do_not_hide_mutating_subcommands(self):
        assert allowlisted_argument_is_unsafe("ruff --isolated format .") is not None
        assert (
            allowlisted_argument_is_unsafe("ruff --config pyproject.toml clean")
            is not None
        )

    def test_ruff_read_only_check_rejects_output_and_source_writes(self):
        assert (
            allowlisted_argument_is_unsafe(
                "ruff check --no-fix --output-file report.json ."
            )
            is not None
        )
        assert (
            allowlisted_argument_is_unsafe("ruff check --no-fix --add-noqa .")
            is not None
        )
