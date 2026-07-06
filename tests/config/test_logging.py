import logging
from unittest.mock import MagicMock, patch

from agent_platform.config.logging import setup_logging


class TestSetupLogging:
    @patch("logging.getLogger")
    def test_default_level(self, mock_get_logger):
        mock_root = MagicMock()
        mock_root.handlers = []
        mock_get_logger.return_value = mock_root

        setup_logging()

        mock_root.setLevel.assert_called_once_with(logging.INFO)
        assert mock_root.addHandler.call_count == 1

    @patch("logging.getLogger")
    def test_debug_level(self, mock_get_logger):
        mock_root = MagicMock()
        mock_root.handlers = []
        mock_get_logger.return_value = mock_root

        setup_logging(logging.DEBUG)

        mock_root.setLevel.assert_called_once_with(logging.DEBUG)

    @patch("logging.getLogger")
    def test_warning_level(self, mock_get_logger):
        mock_root = MagicMock()
        mock_root.handlers = []
        mock_get_logger.return_value = mock_root

        setup_logging(logging.WARNING)

        mock_root.setLevel.assert_called_once_with(logging.WARNING)

    @patch("logging.getLogger")
    def test_error_level(self, mock_get_logger):
        mock_root = MagicMock()
        mock_root.handlers = []
        mock_get_logger.return_value = mock_root

        setup_logging(logging.ERROR)

        mock_root.setLevel.assert_called_once_with(logging.ERROR)

    @patch("logging.getLogger")
    def test_handler_not_added_if_already_exists(self, mock_get_logger):
        mock_root = MagicMock()
        mock_root.handlers = [MagicMock()]
        mock_get_logger.return_value = mock_root

        setup_logging(logging.INFO)

        mock_root.setLevel.assert_called_once()
        mock_root.addHandler.assert_not_called()

    @patch("logging.getLogger")
    def test_formatter_configured(self, mock_get_logger):
        mock_root = MagicMock()
        mock_root.handlers = []
        mock_get_logger.return_value = mock_root

        setup_logging()

        handler = mock_root.addHandler.call_args[0][0]
        assert handler.formatter is not None
        assert "%(asctime)s" in handler.formatter._fmt
