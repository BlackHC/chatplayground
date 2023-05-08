"""Welcome to Pynecone! This file outlines the steps to create a basic app."""
import asyncio
import atexit
import json
import os
import pprint  # noqa: F401
import re
import threading
import time
import types
import typing
import weakref
from contextlib import aclosing
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict

import openai
import pydantic
import pynecone as pc
import watchdog.events
from pydantic.fields import ModelField
from pynecone import Var
from pynecone.components.media.icon import ChakraIconComponent
from pynecone.utils.imports import ImportDict, merge_imports
from pynecone.var import BaseVar, ComputedVar
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class TypedVar(Var):
    """A var of the app state that supports Union types."""

    def __try_type_get(self, name, outer_type_: type):
        # check if type_ is a Union
        if isinstance(outer_type_, types.UnionType):
            for sub_type_ in outer_type_.__args__:
                var = self.__try_type_get(name, sub_type_)
                if var is not None:
                    return var
        if hasattr(outer_type_, "__fields__") and name in outer_type_.__fields__:
            type_ = outer_type_.__fields__[name].outer_type_
            if isinstance(type_, ModelField):
                type_ = type_.type_
            return BaseVar(
                name=f"{self.name}.{name}",
                type_=type_,
                state=self.state,
            )
        return None

    def __getattribute__(self, name: str) -> pc.Var:
        """Get a var attribute.

        Args:
            name: The name of the attribute.

        Returns:
            The var attribute.

        Raises:
            Exception: If the attribute is not found.
        """
        try:
            return super().__getattribute__(name)
        except Exception as e:
            # Check if the attribute is one of the class fields.
            if not name.startswith("_"):
                var = self.__try_type_get(name, self.type_)
                if var is not None:
                    return var

            raise e


Var.register(TypedVar)


class TypedBaseVar(TypedVar, pc.Base):
    """A base (non-computed) var of the app state."""

    # The name of the var.
    name: str

    # The type of the var.
    type_: typing.Any

    # The name of the enclosing state.
    state: str = ""

    # Whether this is a local javascript variable.
    is_local: bool = False

    # Whether this var is a raw string.
    is_string: bool = False

    def __hash__(self) -> int:
        """Define a hash function for a var.

        Returns:
            The hash of the var.
        """
        return hash((self.name, str(self.type_)))

    @typing.no_type_check
    def get_default_value(self) -> typing.Any:
        """Get the default value of the var.

        Returns:
            The default value of the var.
        """
        type_ = self.type_.__origin__ if types.is_generic_alias(self.type_) else self.type_
        if issubclass(type_, str):
            return ""
        if issubclass(type_, types.get_args(typing.Union[int, float])):
            return 0
        if issubclass(type_, bool):
            return False
        if issubclass(type_, list):
            return []
        if issubclass(type_, dict):
            return {}
        if issubclass(type_, tuple):
            return ()
        return set() if issubclass(type_, set) else None

    def get_setter_name(self, include_state: bool = True) -> str:
        """Get the name of the var's generated setter function.

        Args:
            include_state: Whether to include the state name in the setter name.

        Returns:
            The name of the setter function.
        """
        setter = pc.constants.SETTER_PREFIX + self.name
        if not include_state or self.state == "":
            return setter
        return ".".join((self.state, setter))

    def get_setter(self) -> typing.Callable[[pc.State, typing.Any], None]:
        """Get the var's setter function.

        Returns:
            A function that that creates a setter for the var.
        """

        def setter(state: State, value: typing.Any):
            """Get the setter for the var.

            Args:
                state: The state within which we add the setter function.
                value: The value to set.
            """
            setattr(state, self.name, value)

        setter.__qualname__ = self.get_setter_name()

        return setter


BaseVar.register(TypedBaseVar)


# Redefine ComputedVar to reverse the baseclasses
# This fixes https://github.com/pynecone-io/pynecone/issues/907
class TypedComputedVar(TypedVar, property):
    """A field with computed getters."""

    @property
    def name(self) -> str:
        """Get the name of the var.

        Returns:
            The name of the var.
        """
        assert self.fget is not None, "Var must have a getter."
        return self.fget.__name__

    @property
    def type_(self):
        """Get the type of the var.

        Returns:
            The type of the var.
        """
        if "return" in self.fget.__annotations__:
            return self.fget.__annotations__["return"]
        return typing.Any


pc.var = TypedComputedVar
ComputedVar.register(pc.var)


class ReactIcon(ChakraIconComponent):
    """An image icon."""

    tag: str = "Icon"
    as_: Var[pc.EventChain]

    @classmethod
    def create(cls, *children, as_: str, **props):
        """Initialize the Icon component.

        Run some additional checks on Icon component.

        Args:
            children: The positional arguments
            props: The keyword arguments

        Raises:
            AttributeError: The errors tied to bad usage of the Icon component.
            ValueError: If the icon tag is invalid.

        Returns:
            The created component.
        """
        as_var = TypedBaseVar(name=as_, type_=pc.EventChain)
        return super().create(*children, as_=as_var, **props)

    def _get_imports(self) -> ImportDict:
        icon_name = self.as_.name
        # the icon name uses camel case, e.g. BsThreeDotsVertical. The first part (Bs) is the icon sub package
        # convert camel case to snake case
        camel_case = re.sub(r"(?<!^)(?=[A-Z])", "_", icon_name).lower()
        icon_sub_package = camel_case.split("_")[0]
        return merge_imports(super()._get_imports(), {f"react-icons/{icon_sub_package}": {self.as_.name}})


react_icon = ReactIcon.create


class FixedSlider(pc.Slider):
    """A slider that properly handles more complex events.

    (controlled triggers are currently slightly broken)

    TODO: once https://github.com/pynecone-io/pynecone/issues/925 is fixed, this can be removed again.
    """

    @classmethod
    def get_controlled_triggers(cls) -> Dict[str, Var]:
        """No controlled triggers."""
        return {}

    @classmethod
    def get_triggers(cls) -> set[str]:
        """Add on_change_end to the triggers."""
        return {"on_change_end"}


fixed_slider = FixedSlider.create


def float_slider(value: Var[float], setter, min_: float, max_: float, step: float = 0.01):
    """A slider that works with floats (by quantizing them to integers using the given step size)."""

    def on_change_end(final_value: Var[int]):
        event_spec: pc.event.EventSpec = setter(final_value * step + min_)
        return event_spec.copy(update=dict(local_args={final_value.name}))

    return pc.fragment(
        pc.text(BaseVar(name=f"{value.full_name}.toFixed(2)", type_=str)),
        fixed_slider(
            min_=0,
            max_=int((max_ - min_ + step / 2) // step),
            default_value=((value - min_ + step / 2) // step).to(int),
            on_change_end=on_change_end(pc.EVENT_ARG),
        ),
    )


class AutoFocusTextArea(pc.TextArea):
    """Text area that supports the autofocus attribute."""

    auto_focus: Var[typing.Union[str, int, bool]]


auto_focus_text_area = AutoFocusTextArea.create


class UID:
    """A unique ID generator that uses the current time and a counter to generate unique IDs.

    TODO: this is not thread safe, but that should not be a problem for now.
    TODO: ideally we should use a monotonous timer (and encode the timezone as well?)
    """

    _last_time_idx: tuple[str, int] | None = None

    @staticmethod
    def create() -> str:
        """Create a unique ID."""
        now_dt = datetime.now()
        # format is YYYYMMDDHHMMSS
        now_str = now_dt.strftime("%Y%m%d%H%M%S")
        # if we have the same time as last time, increment the counter
        if UID._last_time_idx is not None and UID._last_time_idx[0] == now_str:
            UID._last_time_idx = (now_str, UID._last_time_idx[1] + 1)
        else:
            UID._last_time_idx = (now_str, 0)

        return f"{now_str}#{UID._last_time_idx[1]:04d}"


class SolarizedColors(str, Enum):
    """Solarized colors as HTML hex.

    See https://ethanschoonover.com/solarized/
    """

    base03 = "#002b36"
    base02 = "#073642"
    base01 = "#586e75"
    base00 = "#657b83"
    base0 = "#839496"
    base1 = "#93a1a1"
    base2 = "#eee8d5"
    base3 = "#fdf6e3"
    yellow = "#b58900"
    orange = "#cb4b16"
    red = "#dc322f"
    magenta = "#d33682"
    violet = "#6c71c4"
    blue = "#268bd2"
    cyan = "#2aa198"
    green = "#859900"


class ColorStyle:
    """Color styles for messages."""

    HUMAN = (SolarizedColors.base3, SolarizedColors.base03)
    SYSTEM = (SolarizedColors.base2, SolarizedColors.base03)
    BOTS = [
        SolarizedColors.blue,
        SolarizedColors.yellow,
        SolarizedColors.orange,
        SolarizedColors.magenta,
        SolarizedColors.violet,
        SolarizedColors.cyan,
        SolarizedColors.green,
    ]


class MessageRole(str, Enum):
    """The role of a message in a message thread.

    This is not necessarily the source or origin of the message.
    """

    SYSTEM = "System"
    HUMAN = "Human"
    BOT = "Bot"

    @staticmethod
    def openai_role(role) -> str:
        """Convert a message role to the respective OpenAI role."""
        if role == MessageRole.SYSTEM:
            return "system"
        elif role == MessageRole.HUMAN:
            return "user"
        elif role == MessageRole.BOT:
            return "assistant"
        else:
            raise ValueError(f"Unknown message role {role}")


model_map = {
    "GPT-3.5": "gpt-3.5-turbo",
    "GPT-4": "gpt-4",
}


class MessageSource(pc.Base):
    """The source of a message in a message thread.

    This includes the model type (or None for user generated messages) and the model dict.

    We also store whether a message has been edited or not.
    """

    model_type: str | None = None
    model_dict: dict = {}
    edited: bool = False

    def query_openai(self, messages: list[dict]):
        """Query OpenAI with the given messages using the model_dict and model_type."""
        assert self.model_type is not None
        response = openai.ChatCompletion.acreate(
            messages=messages,
            model=model_map[self.model_type],
            stream=True,
            request_timeout=(20, 60 * 10),
            **self.model_dict,
        )
        return response


class Message(pc.Base):
    """A message in a message thread."""

    uid: str = pydantic.Field(default_factory=UID.create)

    role: MessageRole
    source: MessageSource
    creation_time: int
    content: str | None
    error: str | None = None

    def compute_editable_style(self) -> 'StyledMessage':
        """Compute the style for an editable message."""
        return StyledMessage.create(self, [], [])

    def get_openai_json(self) -> dict:
        """Convert to a JSON object that can be sent to OpenAI."""
        return {
            "role": MessageRole.openai_role(self.role),
            "content": self.content,
        }


class LinkedMessage(pc.Base):
    """A link to a message in a specific message thread.

    This is used for linking alternative messages together across message threads.
    """

    thread_uid: str
    uid: str


class StyledMessage(Message):
    """All the precomputed style information for a message to be rendered in the client."""

    background_color: str
    foreground_color: str
    align: str
    fmt_creation_datetime: str
    fmt_header: str
    prev_linked_messages: list[LinkedMessage]
    next_linked_messages: list[LinkedMessage]

    @classmethod
    def create(
        cls, message: Message, prev_linked_messages: list[LinkedMessage], next_lined_messages: list[LinkedMessage]
    ) -> 'StyledMessage':
        """Create a styled message from a message and its linked (alternative) messages."""
        fmt_creation_datetime = cls.format_datetime(message.creation_time)

        if message.role == MessageRole.SYSTEM:
            background_color, foreground_color = ColorStyle.SYSTEM
            align = "left"
        elif message.role == MessageRole.HUMAN:
            background_color, foreground_color = ColorStyle.HUMAN
            align = "right"
        elif message.role == MessageRole.BOT:
            align = "left"
            if message.source.model_type is not None:
                background_color = ColorStyle.BOTS[
                    list(model_map.keys()).index(message.source.model_type) % len(ColorStyle.BOTS)
                ]
                foreground_color = ColorStyle.HUMAN[1]
            else:
                background_color, foreground_color = ColorStyle.HUMAN
        else:
            raise ValueError(f"Unknown message role {message.role}")

        fmt_header = cls.format_message_header(message.role, message.source)

        return cls(
            align=align,
            background_color=background_color,
            foreground_color=foreground_color,
            fmt_creation_datetime=fmt_creation_datetime,
            fmt_header=fmt_header,
            prev_linked_messages=prev_linked_messages,
            next_linked_messages=next_lined_messages,
            **dict(Message(**dict(message))),
        )

    @staticmethod
    def format_datetime(creation_time: int):
        """Format the creation time of a message in a nice way."""
        # format datetime as "DD.MM.YYYY HH:MM"
        # except if it's yesterday, then just "Yesterday HH:MM"
        # except if it's today, then just "HH:MM"
        # except if it's less than 1 hour ago, then just "MM minutes ago"
        # except if it's less than 1 minute ago, then just "now"
        delta = datetime.now() - datetime.fromtimestamp(creation_time)
        if delta.days > 1:
            return datetime.fromtimestamp(creation_time).strftime("%d.%m.%Y %H:%M")
        elif delta.days == 1:
            return datetime.fromtimestamp(creation_time).strftime("Yesterday %H:%M")
        elif delta.seconds > 3600:
            return datetime.fromtimestamp(creation_time).strftime("%H:%M")
        elif delta.seconds > 60:
            return f"{int(delta.seconds / 60)} minutes ago"
        else:
            return "now"

    @staticmethod
    def format_message_header(role: MessageRole, source: MessageSource):
        """Format the message header in a nice way."""
        if role == MessageRole.SYSTEM:
            fmt_header = 'System'
            if source.model_type is not None:
                fmt_header += f' (by {source.model_type})'
        elif role == MessageRole.HUMAN:
            fmt_header = 'Human'
            if source.model_type is not None:
                fmt_header += f' (by {source.model_type})'
        elif role == MessageRole.BOT:
            if source.model_type is not None:
                fmt_header = source.model_type
            else:
                fmt_header = 'Bot (by Human)'
        else:
            raise ValueError(f'Unknown role: {role}')

        if source.model_type is not None and source.edited:
            fmt_header += " [edited]"

        return fmt_header


class MessageThread(pc.Base):
    """A thread of messages."""

    uid: str = pydantic.Field(default_factory=UID.create)
    messages: list[Message] = pydantic.Field(default_factory=list)

    def get_message_by_uid(self, uid: str) -> Message | None:
        """Get a message by its UID."""
        filtered_message = [message for message in self.messages if message.uid == uid]
        if not filtered_message:
            return None
        assert len(filtered_message) == 1
        return filtered_message[0]


class StyledMessageThread(MessageThread):
    """A thread of messages with precomputed style information."""

    messages: list[StyledMessage]


class MessageGrid(pc.Base):
    """A grid of messages (columns: threads, rows: messages).

    This is used for the expose-style view.
    """

    thread_uids: list[str]
    message_grid: list[list[Message | None]]


class StyledGridCell(pc.Base):
    """A styled cell in a message grid (with information for the client)."""

    message: StyledMessage | None = None
    row_idx: int
    col_idx: int
    thread_uid: str
    col_span: int = 1
    skip: bool = False


class MessageExploration(pc.Base):
    """A bunch of messages that were created together and represent an 'exploration' by the user with LLMs."""

    uid: str = pydantic.Field(default_factory=UID.create)
    title: str = ""
    note: str = ""
    tags: list[str] = pydantic.Field(default_factory=list)

    current_message_thread_uid: str
    message_threads: list[MessageThread]
    # hidden_message_thread_uids: set[str] = pydantic.Field(default_factory=set)

    _message_uid_map: dict[str, Message] = pydantic.PrivateAttr(default_factory=dict)

    def __init__(self, **data):
        """Initialize a message exploration with sensible default values when needed."""
        if 'message_threads' not in data or len(data['message_threads']) == 0:
            data['message_threads'] = [MessageThread()]

        if 'current_message_thread_uid' not in data:
            data['current_message_thread_uid'] = data['message_threads'][0].uid

        super().__init__(**data)
        self._message_uid_map = self._compute_message_uid_map()

    @property
    def message_uid_map(self) -> dict[str, Message]:
        """A map from message UID to message."""
        return self._message_uid_map

    @property
    def current_message_thread(self) -> MessageThread:
        """The current message thread."""
        return self.get_message_thread_by_uid(self.current_message_thread_uid)

    def get_message_thread_by_uid(self, uid: str):
        """Get a message thread by its UID."""
        matches = [mt for mt in self.message_threads if mt.uid == uid]
        assert len(matches) == 1
        return matches[0]

    def insert_message_thread(self, message_thread: MessageThread):
        """Insert a message thread after the current one."""
        # insert the message thread after the current one
        idx = self.message_threads.index(self.current_message_thread)
        self.message_threads.insert(idx + 1, message_thread)
        self.current_message_thread_uid = message_thread.uid

        # update the message uid map
        self._message_uid_map = self._compute_message_uid_map()

    def add_message(self, message: Message):
        """Add a message to the current message thread."""
        # add the message to the current message thread
        self.current_message_thread.messages.append(message)
        self._message_uid_map = self._compute_message_uid_map()

    def delete_message_thread(self, message_thread_uid: str):
        """Delete a message thread."""
        # Remove the message thread with the given uid from the list of message threads.
        idx = next(i for i, mt in enumerate(self.message_threads) if mt.uid == message_thread_uid)
        self.message_threads.pop(idx)

        # Update the current message thread to the previous one if possible.
        if idx > 0:
            self.current_message_thread_uid = self.message_threads[idx - 1].uid
        elif len(self.message_threads) > 0:
            self.current_message_thread_uid = self.message_threads[0].uid
        else:
            self.message_threads = [MessageThread()]
            self.current_message_thread_uid = self.message_threads[0].uid

        # Remove the message thread from the list of hidden message threads if it was hidden.
        # self.hidden_message_thread_uids.discard(message_thread_uid)

        # update the message uid map
        self._message_uid_map = self._compute_message_uid_map()

    # def set_message_thread_visible(self, message_thread_uid: str, visible: bool):
    #     if visible:
    #         self.hidden_message_thread_uids.discard(message_thread_uid)
    #     else:
    #         self.hidden_message_thread_uids.add(message_thread_uid)

    def compute_message_grid(self) -> MessageGrid:
        """Get a grid of messages (columns: threads, rows: messages)."""
        # for each message thread create a tuple of all message uids in the thread
        message_threads_map = {
            tuple(message.uid for message in message_thread.messages): message_thread
            for message_thread in self.message_threads
        }
        # sort lexicographically by message uid tuples
        sorted_message_threads_items = sorted(list(message_threads_map.items()), key=lambda kv: kv[0])

        # create grid
        row_count = max(len(message_thread.messages) for message_thread in self.message_threads)
        message_grid = [[None for _ in message_threads_map] for _ in range(row_count)]
        message_thread_uids = []
        for col, (_, message_thread) in enumerate(sorted_message_threads_items):
            message_thread_uids.append(message_thread.uid)
            for row, message in enumerate(message_thread.messages):
                message_grid[row][col] = message  # type: ignore

        return MessageGrid(thread_uids=message_thread_uids, message_grid=message_grid)

    def _grid_deduplicate_message_threads(self, message_threads, row):
        """Deduplicate message threads by their message at the given row.

        This is used to create deduplicated alternative messages.
        """
        cleaned_message_threads = []
        message_uids = set()
        for message_thread in message_threads:
            if len(message_thread.messages) <= row:
                continue
            message_uid = message_thread.messages[row].uid
            if message_uid in message_uids:
                continue
            message_uids.add(message_uid)
            cleaned_message_threads.append(message_thread)
        return cleaned_message_threads

    def compute_styled_current_message_thread(self) -> StyledMessageThread:
        """Compute the current message thread with styling information."""
        current_message_thread = self.current_message_thread

        message_thread_beam = self.message_threads

        styled_messages = []
        for row, current_message in enumerate(current_message_thread.messages):
            # find all message threads in the beam that have a message with the same uid in the same row (if available)
            new_message_thread_beam = []
            for message_thread in message_thread_beam:
                if len(message_thread.messages) > row and message_thread.messages[row].uid == current_message.uid:
                    new_message_thread_beam.append(message_thread)
            # split message thread beam into three parts:
            # 1. message threads that are also in the new message thread beam, and
            #    message threads that are not in the new message thread beam:
            #    2. message threads that come before the current message thread
            #    3. message threads that come after the current message thread

            message_threads = [mt for mt in message_thread_beam]
            message_threads = self._grid_deduplicate_message_threads(message_threads, row)

            # obtain the index of the current message thread in the new message thread beam
            current_message_idx = next(
                i for i, mt in enumerate(message_threads) if mt.messages[row].uid == current_message.uid
            )

            linked_messages_before: list[LinkedMessage] = [
                LinkedMessage(thread_uid=message_thread.uid, uid=message_thread.messages[row].uid)
                for message_thread in message_threads[:current_message_idx]
            ]
            linked_messages_after: list[LinkedMessage] = [
                LinkedMessage(thread_uid=message_thread.uid, uid=message_thread.messages[row].uid)
                for message_thread in message_threads[current_message_idx + 1 :]
            ]
            styled_message = StyledMessage.create(current_message, linked_messages_before, linked_messages_after)
            styled_messages.append(styled_message)

            message_thread_beam = new_message_thread_beam

        return StyledMessageThread(
            uid=current_message_thread.uid,
            messages=styled_messages,
        )

    def _compute_message_uid_map(self):
        message_uid_map = {}
        for message_thread in self.message_threads:
            for message in message_thread.messages:
                stored_message = message_uid_map.setdefault(message.uid, message)
                assert stored_message == message
        return message_uid_map


def render_edit_thread_button(message: StyledMessage):
    """Render the edit thread button."""
    return pc.button(
        pc.icon(tag="edit"),
        size='xs',
        variant='ghost',
        on_click=lambda: EditableMessageState.enter_editing(message.uid),  # type: ignore
    )


def render_fork_thread_button(message: StyledMessage):
    """Render the fork thread button."""
    return pc.button(
        pc.icon(tag="repeat_clock"),
        size='xs',
        variant='ghost',
        on_click=lambda: EditableMessageState.enter_forking(message.uid),  # type: ignore
    )


@typing.no_type_check
def render_message_toolbar(message: StyledMessage):
    """Render the message toolbar.

    The toolbar contains the menu, edit, and fork buttons as well as the navigation buttons for alternative messages.
    """
    num_threads = 1 + message.next_linked_messages.length() + message.prev_linked_messages.length()
    thread_index = 1 + message.prev_linked_messages.length()
    is_not_bot = message.role != "Bot"
    return pc.fragment(
        pc.cond(
            message.uid == EditableMessageState.editable_message_uid,
            pc.hstack(
                pc.button(
                    pc.icon(tag="check"), size='xs', variant='ghost', on_click=EditableMessageState.submit_editing
                ),
                pc.divider(orientation="vertical", height="1em"),
                pc.button(
                    pc.icon(tag="close"), size='xs', variant='ghost', on_click=EditableMessageState.cancel_editing
                ),
                pc.divider(orientation="vertical", height="1em"),
            ),
            pc.fragment(
                render_message_menu(message),
                pc.cond(is_not_bot, render_fork_thread_button(message), render_edit_thread_button(message)),
            ),
        ),
        pc.fragment(
            pc.button(
                pc.icon(tag="chevron_left"),
                size='xs',
                variant='ghost',
                is_disabled=EditableMessageState.is_editing | (message.prev_linked_messages.length() == 0),
                on_click=lambda: NavigationState.go_to_message_thread(
                    message.prev_linked_messages[-1].thread_uid,
                    message.prev_linked_messages[-1].uid,
                ),
            ),
            pc.button(
                react_icon(as_="ImTree"),
                size='xs',
                variant='ghost',
                is_disabled=(
                    EditableMessageState.is_editing
                    | ((message.next_linked_messages.length() + message.prev_linked_messages.length()) == 0)
                ),
                on_click=MessageOverviewState.show_alternatives(message.uid),
            ),
            pc.button(
                pc.icon(tag="chevron_right"),
                size='xs',
                variant='ghost',
                is_disabled=EditableMessageState.is_editing | (message.next_linked_messages.length() == 0),
                on_click=lambda: NavigationState.go_to_message_thread(
                    message.next_linked_messages[0].thread_uid,
                    message.next_linked_messages[0].uid,
                ),
            ),
        ),
        pc.cond(
            num_threads > 1,
            pc.fragment(
                pc.text(thread_index),
                pc.text("/", color=SolarizedColors.base0),
                pc.text(num_threads, color=SolarizedColors.base0),
            ),
        ),
    )


def render_carousel_message(message: StyledMessage):
    """Render a message in the carousel of alternative messages."""
    return pc.grid_item(
        pc.button(
            pc.vstack(
                pc.box(
                    pc.flex(
                        pc.text(message.fmt_header),
                        pc.spacer(),
                        pc.text(message.fmt_creation_datetime),
                        width="100%",
                    ),
                    background_color=message.background_color,
                    color=message.foreground_color,
                    border_radius="15px 15px 0 0",
                    width="100%",
                    padding="0.25em 0.25em 0 0.5em",
                    margin_bottom="0",
                ),
                pc.cond(
                    message.error,
                    pc.fragment(
                        pc.cond(
                            message.content,
                            pc.box(
                                render_markdown(message.content),
                                border_radius="0 0 0 0",
                                color=message.foreground_color,
                                border_width="medium",
                                border_color=message.foreground_color,
                                width="100%",
                                padding="0 0.25em 0 0.25em",
                            ),
                        ),
                        pc.box(
                            render_markdown(message.error),
                            border_radius="0 0 15px 15px",
                            color=message.foreground_color,
                            background_color=SolarizedColors.red,
                            width="100%",
                            padding="0 0.25em 0 0.25em",
                        ),
                    ),
                    pc.cond(
                        message.content,
                        pc.box(
                            render_markdown(message.content),
                            border_radius="0 0 15px 15px",
                            color=message.foreground_color,
                            border_width="medium",
                            border_color=message.foreground_color,
                            width="100%",
                            padding="0 0.25em 0 0.25em",
                        ),
                    ),
                ),
                width="30ch",
                max_width="30ch",
                padding_top="0.5em",
                padding_bottom="0.5em",
                spacing="0em",
            ),
            width="30ch",
            max_width="32ch",
            height="100%",
            on_click=lambda: MessageOverviewState.hide_alternatives(message.uid),  # type: ignore
            variant="solid",
        ),
        row_start=1,
    )


def render_markdown(content: str | None, **props):
    """Render a markdown string using sensible defaults for the styles."""
    return pc.box(
        pc.cond(content, pc.markdown(content), pc.text("")),
        max_height="50vh",
        style={"word-wrap": "break-word", "white-space": "pre-wrap", "text-align": "left", "overflow": "scroll"},
        **props,
    )


def render_message_overview_carousel():
    """Render the carousel of alternative messages."""
    return pc.flex(
        pc.box(
            pc.grid(
                pc.foreach(MessageOverviewState.styled_previous_messages, render_carousel_message),
                render_carousel_message(MessageOverviewState.styled_current_message),
                pc.foreach(MessageOverviewState.styled_next_messages, render_carousel_message),
                gap="1ch",
                width="fit-content",
                zoom="0.75",
            ),
            max_width="100%",
            overflow="scroll",
        ),
        justify_content="center",
        width="100vw",
        max_width="100vw",
        margin_left="50%",
        transform="translateX(-50%)",
        padding_bottom="0.5em",
    )


@typing.no_type_check
def render_grid_cell(styled_grid_cell: StyledGridCell):
    """Render a grid cell in the expose-style grid."""
    message = styled_grid_cell.message

    return pc.cond(
        ~styled_grid_cell.skip,
        pc.grid_item(
            pc.cond(
                message,
                pc.vstack(
                    pc.box(
                        pc.flex(
                            pc.text(message.fmt_header),
                            pc.spacer(),
                            pc.text(message.fmt_creation_datetime),
                            width="100%",
                        ),
                        background_color=message.background_color,
                        color=message.foreground_color,
                        border_radius="15px 15px 0 0",
                        width="100%",
                        padding="0.25em 0.25em 0 0.5em",
                        margin_bottom="0",
                    ),
                    pc.cond(
                        message.error,
                        pc.fragment(
                            pc.cond(
                                message.content,
                                pc.box(
                                    render_markdown(message.content),
                                    border_radius="0 0 0 0",
                                    color=message.foreground_color,
                                    border_width="medium",
                                    border_color=message.foreground_color,
                                    width="100%",
                                    padding="0 0.25em 0 0.25em",
                                    style={"z-index": 2},
                                ),
                            ),
                            pc.box(
                                render_markdown(message.error),
                                border_radius="0 0 15px 15px",
                                color=message.foreground_color,
                                background_color=SolarizedColors.red,
                                width="100%",
                                padding="0 0.25em 0 0.25em",
                            ),
                        ),
                        pc.cond(
                            message.content,
                            pc.box(
                                render_markdown(message.content),
                                border_radius="0 0 15px 15px",
                                color=message.foreground_color,
                                border_width="medium",
                                border_color=message.foreground_color,
                                width="100%",
                                padding="0 0.25em 0 0.25em",
                                style={"z-index": 2},
                            ),
                        ),
                    ),
                    width="30ch",
                    max_width="30ch",
                    padding="0.5em",
                    spacing="0em",
                ),
            ),
            row_start=styled_grid_cell.row_idx + 1,
            col_start=styled_grid_cell.col_idx,
            col_span=styled_grid_cell.col_span,
            justify_self="center",
            style={
                "border-width": "0 thin 0 thin",
                "border_color": SolarizedColors.base1,
                "width": "100%",
                "justify-content": "center",
                "display": "flex",
            },
        ),
    )


def render_grid_row(message_row: list[StyledGridCell]):
    """Render a row in the expose-style grid."""
    return pc.foreach(message_row, render_grid_cell)


def render_grid_column_button(thread_uid: str, index):
    """Render a button to remove a thread from the grid and a big button that covers the whole column to 'zoom in'."""
    return pc.fragment(
        pc.grid_item(
            pc.button(pc.icon(tag="delete"), variant="outline", margin="1em"),
            row_start=1,
            row_span=1,
            col_start=index.to(int) + 1,
            col_span=1,
            on_click=lambda: MessageGridState.remove_thread(thread_uid),  # type: ignore
        ),
        pc.grid_item(
            pc.button(
                width="100%",
                height="100%",
                on_click=lambda: MessageGridState.go_to_thread(thread_uid),  # type: ignore
                variant="ghost",
                opacity=0.5,
                minWidth="30ch",
                min_height="30ch",
            ),
            row_start=2,
            row_span=MessageGridState.styled_grid_cells.length(),
            col_start=index.to(int) + 1,
            col_span=1,
            style={"z-index": 1},
        ),
    )


def render_message_grid():
    """Render the expose-style grid."""
    return pc.flex(
        pc.box(
            pc.grid(
                pc.foreach(MessageGridState.column_thread_uids, render_grid_column_button),
                pc.foreach(MessageGridState.styled_grid_cells, render_grid_row),
                width="fit-content",
                zoom="0.75",
            ),
            max_width="100%",
            overflow="scroll",
        ),
        justify_content="center",
        width="100vw",
        max_width="100vw",
        margin_left="50%",
        transform="translateX(-50%)",
        padding_bottom="0.5em",
    )


def render_message_normal(message: StyledMessage):
    """Render a message in the normal mode of interaction (no active LLM request)."""
    return pc.hstack(
        pc.vstack(
            pc.box(
                pc.flex(
                    render_message_toolbar(message),
                    pc.spacer(),
                    pc.text(message.fmt_header),
                    pc.cond(
                        message.role == "System",
                        pc.button(pc.icon(tag="info"), size="xs", variant="outline"),
                    ),
                    pc.spacer(),
                    pc.text(message.fmt_creation_datetime),
                    width="100%",
                ),
                background_color=message.background_color,
                color=message.foreground_color,
                border_radius="15px 0 0 0",
                width="100%",
                padding="0.25em 0.25em 0.25em 0.5em",
                margin_bottom="0",
            ),
            pc.cond(
                message.error,
                pc.fragment(
                    pc.cond(
                        message.content,
                        pc.box(
                            render_markdown(message.content),
                            border_radius="0 0 0 0",
                            color=message.foreground_color,
                            border_width="medium",
                            border_color=message.foreground_color,
                            width="100%",
                            padding="0 0.25em 0 0.25em",
                        ),
                    ),
                    pc.box(
                        render_markdown(message.error),
                        border_radius="0 0 15px 0",
                        color=message.foreground_color,
                        background_color=SolarizedColors.red,
                        width="100%",
                        padding="0 0.25em 0 0.25em",
                    ),
                ),
                pc.cond(
                    message.content,
                    pc.box(
                        render_markdown(message.content),
                        border_radius="0 0 15px 0",
                        color=message.foreground_color,
                        border_width="medium",
                        border_color=message.foreground_color,
                        width="100%",
                        padding="0 0.25em 0 0.25em",
                    ),
                ),
            ),
            width="60ch",
            padding_bottom="0.5em",
            spacing="0em",
        ),
        justify_content=message.align,
        width="100%",
    )


def render_message_active_request(message: StyledMessage):
    """Render a message in the active mode of interaction (active LLM request)."""
    return pc.hstack(
        pc.vstack(
            pc.box(
                pc.flex(
                    pc.spacer(),
                    pc.text(message.fmt_header),
                    pc.spacer(),
                    pc.text(message.fmt_creation_datetime),
                    width="100%",
                ),
                background_color=message.background_color,
                color=message.foreground_color,
                border_radius="15px 0 0 0",
                width="100%",
                padding="0.25em 0.25em 0.25em 0.5em",
                margin_bottom="0",
            ),
            pc.cond(
                message.error,
                pc.fragment(
                    pc.cond(
                        message.content,
                        pc.box(
                            render_markdown(message.content),
                            border_radius="0 0 0 0",
                            color=message.foreground_color,
                            border_width="medium",
                            border_color=message.foreground_color,
                            width="100%",
                            padding="0 0.25em 0 0.25em",
                        ),
                    ),
                    pc.box(
                        render_markdown(message.error),
                        border_radius="0 0 15px 0",
                        color=message.foreground_color,
                        background_color=SolarizedColors.red,
                        width="100%",
                        padding="0 0.25em 0 0.25em",
                    ),
                ),
                pc.cond(
                    message.content,
                    pc.box(
                        render_markdown(message.content),
                        border_radius="0 0 15px 0",
                        color=message.foreground_color,
                        border_width="medium",
                        border_color=message.foreground_color,
                        width="100%",
                        padding="0 0.25em 0 0.25em",
                    ),
                ),
            ),
            width="60ch",
            padding_bottom="0.5em",
            spacing="0em",
        ),
        justify_content=message.align,
        width="100%",
    )


def render_editable_message(message: StyledMessage):
    """Render a message that we currently edit (either for forking or for editing)."""
    return pc.hstack(
        pc.vstack(
            pc.center(
                pc.flex(
                    render_message_toolbar(message),
                    pc.spacer(),
                    pc.box(
                        pc.select(
                            [role.value for role in MessageRole],
                            default_value=message.role,
                            # on_change=SelectState.set_option,
                            variant="unstyled",
                            on_change=EditableMessageState.set_editable_message_role,
                        ),
                        width='30%',
                    ),
                    pc.spacer(),
                    pc.text(message.fmt_creation_datetime),
                    width="100%",
                ),
                background_color=message.background_color,
                color=message.foreground_color,
                border_radius="15px 0 0 0",
                width="100%",
                padding="0.25em 0.25em 0 0.5em",
                margin_bottom="0",
            ),
            pc.cond(
                message.error,
                pc.fragment(
                    render_editable_message_content(message, with_error=False), render_editable_message_error(message)
                ),
                render_editable_message_content(message, with_error=False),
            ),
            width="75%",
            padding_bottom="0.5em",
            spacing="0em",
        ),
        justify_content=message.align,
        width="100%",
    )


def render_editable_message_error(message):
    """Render the error message of an editable message."""
    return pc.box(
        pc.flex(
            pc.box(
                render_markdown(message.error),
            ),
            pc.spacer(),
            pc.button(
                pc.icon(tag="delete"),
                size='xs',
                variant='ghost',
                on_click=EditableMessageState.clear_editable_message_error,
            ),
            height="100%",
        ),
        border_radius="0 0 15px 0",
        color=message.foreground_color,
        background_color=SolarizedColors.red,
        width="100%",
        padding="0 0.25em 0 0.25em",
    )


def render_editable_message_content(message, with_error: bool):
    """Render the content of an editable message."""
    return auto_focus_text_area(
        auto_focus=True,
        default_value=message.content,
        border_radius="0 0 0 0" if with_error else "0 0 15px 0",
        color=message.foreground_color,
        border_width="medium",
        width="100%",
        padding="0 0.25em 0 0.25em",
        on_blur=EditableMessageState.set_editable_message_content,
    )


def render_message_menu(message: StyledMessage):
    """Render the menu of a message."""
    return pc.popover(
        pc.popover_trigger(
            pc.button(pc.icon(tag="hamburger"), size='xs', variant='ghost', is_disabled=EditableMessageState.is_editing)
        ),
        pc.popover_content(
            pc.popover_arrow(),
            pc.popover_body(
                pc.hstack(
                    # pc.button(pc.icon(tag="add"), size='xs', variant='ghost'),
                    pc.button(
                        pc.icon(tag="delete"),
                        size='xs',
                        variant='ghost',
                        on_click=EditableMessageState.delete_message(message.uid),
                    ),
                    pc.divider(orientation="vertical", height="1em"),
                    render_edit_thread_button(message),
                    render_fork_thread_button(message),  # type: ignore
                )
            ),
            pc.popover_close_button(),
            width='10em',  # '12em',
        ),
        trigger="hover",
    )


def render_message(message: StyledMessage):
    """Render a message (either in edit/fork mode, or in normal mode, or as carousel that includes alternatives)."""
    return pc.cond(
        message.uid == EditableMessageState.editable_message_uid,
        render_editable_message(EditableMessageState.styled_editable_message),
        pc.cond(
            message.uid == MessageOverviewState.current_message_uid,
            render_message_overview_carousel(),
            render_message_normal(message),
        ),
    )


#
# def render_message_thread_menu(message_thread: MessageThread):
#     return
#         pc.popover(
#         pc.popover_trigger(pc.button(pc.icon(tag="hamburger"), size='xs', variant='ghost')),
#         pc.popover_content(
#             pc.popover_arrow(),
#             pc.popover_body(
#                 pc.hstack(
#                     pc.button(react_icon(as_="MdViewSidebar"), size='xs', variant='ghost'),
#                     pc.button(pc.icon(tag="close"), size='xs', variant='ghost'),
#                     pc.button(pc.icon(tag="copy"), size='xs', variant='ghost'),
#                     pc.button(pc.icon(tag="delete"), size='xs', variant='ghost'),
#                     # pc.divider(orientation="vertical", height="1em"),
#                     # pc.button(pc.icon(tag="lock"), size='xs', variant='ghost'),
#                     # pc.button(pc.icon(tag="edit"), size='xs', variant='ghost'),
#                 )
#             ),
#             pc.popover_close_button(),
#             width='17em',
#         ),
#         trigger="hover",
#     )


def render_app_drawer():
    """Render the sidebar that contains all the explorations we have saved."""

    def render_exploration_buttons(message_exploration: MessageExploration):
        return pc.hstack(
            pc.button(
                pc.cond(
                    message_exploration.title != "",
                    pc.text(message_exploration.title),
                    pc.text("Untitled", as_="i", color=SolarizedColors.base1),
                ),
                width="100%",
                variant=pc.cond(message_exploration.uid == State.message_exploration_uid, "solid", "outline"),
                on_click=State.select_message_exploration(message_exploration.uid),  # type: ignore
            ),
            pc.button(
                pc.icon(tag="delete"),
                size='xs',
                variant='ghost',
                on_click=RemoveExplorationModalState.ask_to_remove(message_exploration.uid),  # type: ignore
            ),
        )

    return pc.drawer(
        pc.drawer_overlay(
            pc.drawer_content(
                pc.drawer_header(
                    pc.hstack(
                        pc.text("ChatPlayground"),
                        pc.spacer(),
                        pc.button(
                            pc.icon(tag="close"),
                            on_click=State.set_drawer_visible(False),
                            float="right",
                            size='xs',
                            variant='ghost',
                        ),
                    )
                ),
                pc.drawer_body(
                    pc.divider(margin="0.5em"),
                    pc.vstack(
                        pc.foreach(State.get_available_message_explorations, render_exploration_buttons),
                        spacing="0.5em",
                        overflow="auto",
                        height="100%",
                        display="block",
                    ),
                ),
                pc.modal(
                    pc.modal_overlay(
                        pc.modal_content(
                            pc.modal_header("Confirm"),
                            pc.modal_body(
                                "Do you want to delete '"
                                + RemoveExplorationModalState.remove_modal_exploration_title
                                + "'?"
                            ),
                            pc.modal_footer(
                                pc.button(
                                    "Yes",
                                    on_click=RemoveExplorationModalState.do_remove,
                                    color_scheme="red",
                                    margin="0.5em",
                                ),
                                pc.button(
                                    "No",
                                    on_click=RemoveExplorationModalState.cancel,
                                    auto_focus=True,
                                    variant="ghost",
                                ),
                            ),
                        )
                    ),
                    is_open=RemoveExplorationModalState.show_modal,
                ),
                # pc.drawer_footer(
                #
                # ),
                # bg="rgba(0, 0, 0, 0.3)",
            )
        ),
        is_open=State.drawer_visible,
        close_on_overlay_click=True,
        placement="left",
        auto_focus=True,
        close_on_esc=True,
    )


def render_note_editor():
    """Render the note editor."""
    return pc.cond(
        ~State.note_editor,
        pc.button(
            pc.flex(
                pc.cond(
                    State.note,
                    render_markdown(State.note),
                    pc.markdown("Add a note here...", color=SolarizedColors.base1),
                ),
                pc.spacer(),
                pc.icon(tag="edit", size="xs"),
                width="100%",
                style={"font-weight": "normal"},
            ),
            on_click=State.start_note_editing(None),
            variant="ghost",
            width="100%",
            padding="0.25em",
        ),
        pc.editable(
            pc.editable_preview(),
            pc.editable_textarea(),
            start_with_edit_view=True,
            default_value=State.note,
            on_submit=State.update_note,
            on_cancel=State.cancel_note_editing,
            margin="0.25em",
        ),
    )


def render_model_options():
    """Render the model options."""
    return pc.table_container(
        pc.table(
            pc.tbody(
                pc.tr(
                    pc.th("Model"),
                    pc.td(
                        pc.select(
                            [
                                "GPT-3.5",
                                "GPT-4",
                            ],
                            default_value=ModelRequestsState.model_type,
                            on_change=ModelRequestsState.set_model_type,
                        ),
                        style={"min-width": "20em"},
                    ),
                ),
                pc.tr(
                    pc.th("Temperature"),
                    pc.td(
                        float_slider(
                            ModelRequestsState.temperature,
                            ModelRequestsState.set_temperature,
                            min_=0,
                            max_=1,
                            step=0.01,
                        ),
                    ),
                ),
                pc.tr(
                    pc.th("Max Tokens"),
                    pc.td(
                        pc.text(ModelRequestsState.max_tokens),
                        pc.slider(
                            min_=0,
                            max_=2048,
                            default_value=ModelRequestsState.max_tokens,
                            on_change_end=ModelRequestsState.set_max_tokens,
                        ),
                    ),
                ),
                pc.tr(
                    pc.th("Top P"),
                    pc.td(
                        float_slider(ModelRequestsState.top_p, ModelRequestsState.set_top_p, min_=0, max_=1, step=0.01),
                    ),
                ),
                pc.tr(
                    pc.th("Frequency Penalty"),
                    pc.td(
                        float_slider(
                            ModelRequestsState.frequency_penalty,
                            ModelRequestsState.set_frequency_penalty,
                            min_=0,
                            max_=2,
                            step=0.01,
                        ),
                    ),
                ),
                pc.tr(
                    pc.th("Presence Penalty"),
                    pc.td(
                        float_slider(
                            ModelRequestsState.presence_penalty,
                            ModelRequestsState.set_presence_penalty,
                            min_=0,
                            max_=1,
                            step=0.01,
                        ),
                    ),
                ),
            ),
            width="50%",
        ),
    )


def render_request_ui():
    """Render the request UI.

    The request UI contains the model options and the request button (and the chat UI if chat mode is enabled).
    """
    return pc.cond(
        ~ModelRequestsState.active_generation,
        pc.fragment(
            pc.divider(margin="0.5em"),
            pc.button(
                "Request Continuation",
                on_click=ModelRequestsState.request_chat_model_completion,
            ),
            pc.cond(
                ModelRequestsState.chat_mode,
                pc.fragment(
                    pc.divider(margin="0.5em"),
                    pc.hstack(
                        pc.text_area(
                            placeholder="Write a message...",
                            default_value=ModelRequestsState.user_message_text,
                            width="100%",
                            min_height="5em",
                            on_blur=ModelRequestsState.set_user_message_text,
                        ),
                        pc.button(
                            react_icon(as_="TbSend"),
                            size="lg",
                            variant="outline",
                            float="right",
                            on_click=ModelRequestsState.submit_message,
                        ),
                    ),
                ),
            ),
        ),
        pc.fragment(
            pc.button(
                pc.spinner(color="blue", size="xs"),
                pc.text("Cancel", margin_left="1em", on_click=ModelRequestsState.cancel_active_chat_completion),
            ),
        ),
    )


def render_message_exploration(message_thread: MessageThread):
    """Render the overall message exploration UI which includes its title."""
    return pc.box(
        pc.hstack(
            pc.fragment(
                pc.button(
                    react_icon(as_="RiSideBarFill"),
                    size='xs',
                    variant=pc.cond(State.drawer_visible, 'solid', 'ghost'),
                    on_click=State.set_drawer_visible(~State.drawer_visible),
                ),
                pc.button(
                    react_icon(as_="RiChatNewLine"), variant="ghost", size="xs", on_click=State.new_message_exploration
                ),
                pc.button(
                    react_icon(as_="BsFillGrid3X3GapFill"),
                    size='xs',
                    variant=pc.cond(MessageGridState.is_grid_visible, 'solid', 'ghost'),
                    on_click=MessageGridState.toggle_grid,
                ),
            ),
            pc.heading(
                pc.editable(
                    pc.editable_preview(),
                    pc.editable_input(),
                    value=State.title_text,
                    placeholder="Untitled",
                    on_change=State.set_title_text,
                    on_submit=State.update_title,
                ),
                size="md",
            ),
        ),
        pc.divider(margin="0.5em"),
        pc.box(
            render_note_editor(),
            padding="0.5em",
        ),
        pc.divider(margin="0.5em"),
        pc.cond(
            MessageGridState.is_grid_visible,
            render_message_grid(),
            pc.box(
                pc.cond(
                    ModelRequestsState.active_generation,
                    pc.foreach(message_thread.messages, render_message_active_request),
                    pc.foreach(message_thread.messages, render_message),
                ),
                render_request_ui(),
                pc.box(
                    pc.accordion(
                        pc.accordion_item(
                            pc.accordion_button(
                                pc.hstack(
                                    pc.heading("Model Options", size="sm"),
                                    pc.accordion_icon(),
                                    min_width="100%",
                                ),
                            ),
                            pc.accordion_panel(
                                render_model_options(),
                            ),
                        ),
                    ),
                    margin_top="0.5em",
                ),
            ),
        ),
        width="100%",
    )


@dataclass
class MessageExplorationStorage:
    """A dataclass that stores a message exploration and the last time it was updated."""

    message_exploration: MessageExploration
    last_updated: float


class StoredExplorationsSingleton(FileSystemEventHandler):
    """Singleton class that manages all message explorations in the file system.

    This class is responsible for:
    - loading all message explorations from the file system
    - saving all message explorations to the file system
    - sending update events to all registered states
    """

    DIRECTORY: str = "./message_threads"
    observer: Observer = Observer()
    _stored_explorations: dict[str, MessageExplorationStorage] = {}

    _receivers: weakref.WeakValueDictionary[str, pc.State] = weakref.WeakValueDictionary()

    lock: threading.Lock = threading.Lock()

    def __init__(self):
        """Initialize the singleton."""
        self.setup_watchdog()
        self.load_all()

    def get_last_updated_timestamp(self, message_exploration_uid: str):
        """Get the last updated timestamp of a message exploration."""
        # find the message exploration
        for _, stored_exploration in self._stored_explorations.items():
            if stored_exploration.message_exploration.uid == message_exploration_uid:
                return stored_exploration.last_updated
        return 0

    def register_state(self, state: pc.State):
        """Register a state to receive update events.

        We use this to keep track of active clients using a weakref value dictionary.
        """
        main_state: pc.State = state.parent_state if state.parent_state else state
        # TODO: check that we have exactly one state per session id?
        if main_state.get_sid() not in self._receivers:
            self._receivers[main_state.get_sid()] = main_state

    def send_update_event(self, origin: pc.State | None, message_exploration_uid: str):
        """Send an update event to all registered states (except the one that triggered the update).

        TODO: we should create a single background thread that sends events to all registered states and not
            create a new thread for each state.
        """
        if origin is not None:
            origin = origin.parent_state if origin.parent_state else origin

        async def send_events():
            # send an event
            for state in self._receivers.values():
                if state.get_sid() != origin.get_sid():
                    # noinspection PyNoneFunctionAssignment,PyArgumentList
                    event_handler: pc.event.EventHandler = State.on_message_exploration_update(  # type: ignore
                        message_exploration_uid  # type: ignore
                    )
                    await send_event(state, event_handler)

        # app.sio.start_background_task(send_event, state, event_handler)
        helper_thread(send_events)

    def get(self, state: pc.State, uid: str):
        """Get a message exploration by its uid."""
        self.register_state(state)
        for exploration_storage in self._stored_explorations.values():
            if (exploration := exploration_storage.message_exploration).uid == uid:
                return exploration.copy(deep=True)
        raise KeyError(f"MessageExploration with uid {uid} not found")

    def get_all(self, state: pc.State) -> list[MessageExploration]:
        """Get all message explorations."""
        self.register_state(state)

        # deep copy all
        return [
            stored_exploration.message_exploration.copy(deep=True)
            for stored_exploration in self._stored_explorations.values()
        ]

    @staticmethod
    def to_json(message_exploration: MessageExploration):
        """Convert a message exploration to json."""
        return pydantic.BaseModel.json(message_exploration, indent=1)

    def remove(self, state: pc.State, message_exploration_uid: str):
        """Remove a message exploration."""
        # find the file name in the _explorations (by uid)
        # then remove the file
        for filename, stored in self._stored_explorations.items():
            if stored.message_exploration.uid == message_exploration_uid:
                path = os.path.join(self.DIRECTORY, filename)
                # check that path exists as is not below the directory
                if not os.path.exists(path) or not os.path.samefile(
                    self.DIRECTORY, os.path.commonpath([path, self.DIRECTORY])
                ):
                    raise ValueError(f"Invalid path {path}")
                os.remove(path)
                # remove from the dict, too
                del self._stored_explorations[filename]

                self.send_update_event(state, message_exploration_uid)
                break
        else:
            raise KeyError(f"MessageExploration with uid {message_exploration_uid} not found")

    @staticmethod
    def convert_title_to_filename_part(title: str):
        """Convert a title to a valid filename part."""
        # remove characters that are not valid in file names
        title = re.sub(r"[^a-zA-Z0-9_ ]", "", title)
        # replace spaces with single underscores and underscores with double underscores
        title = re.sub(r"_+", "__", title)
        title = re.sub(r" +", "_", title)
        return title.replace(" ", "_").lower()

    @staticmethod
    def get_canonical_prefix_and_filename(message_exploration: MessageExploration):
        """Get a canonical prefix and filename for a message exploration."""
        if message_exploration.title:
            prefix = StoredExplorationsSingleton.convert_title_to_filename_part(message_exploration.title)
            filename = f"{prefix}_{time.time()}.json"
        else:
            prefix = message_exploration.uid
            filename = f"{prefix}.json"

        return prefix, filename

    def update(self, state: pc.State, message_exploration: MessageExploration):
        """Update a message exploration from the UI and write it to disk.

        Send an update event to all registered states as necessary.
        """
        with self.lock:
            self.register_state(state)

            # find the file name in the _explorations (by uid)
            # then update the dict and then write the file

            for filename, storage in self._stored_explorations.items():
                if (stored_exploration := storage.message_exploration).uid == message_exploration.uid:
                    # check if anything changed
                    if stored_exploration != message_exploration:
                        # get a temporary file name based on the original file name and the current time
                        temp_filename = f"{filename}.{time.time()}.tmp"
                        temp_path = os.path.join(self.DIRECTORY, temp_filename)
                        with open(temp_path, 'wt') as f:
                            f.write(self.to_json(message_exploration))
                            # make sure that all data is on disk
                            # see http://stackoverflow.com/questions/7433057/is-rename-without-fsync-safe
                            f.flush()
                            os.fsync(f.fileno())

                        target_path = os.path.join(self.DIRECTORY, filename)
                        os.replace(temp_path, target_path)

                        # do we want to rename the file?
                        canonical_prefix, canonical_filename = self.get_canonical_prefix_and_filename(
                            message_exploration
                        )
                        if not filename.startswith(canonical_prefix):
                            print(f"Renaming {filename} to {canonical_filename}")
                            # rename the file
                            new_path = os.path.join(self.DIRECTORY, canonical_filename)
                            os.rename(target_path, new_path)

                            del self._stored_explorations[filename]
                            filename = canonical_filename

                        # update the dict
                        self._stored_explorations[filename] = MessageExplorationStorage(
                            message_exploration=message_exploration.copy(deep=True), last_updated=time.time()
                        )

                        self.send_update_event(state, message_exploration.uid)

                        return True
                    break
            else:
                # Otherwise, create a new file
                _, filename = self.get_canonical_prefix_and_filename(message_exploration)
                path = os.path.join(self.DIRECTORY, filename)
                with open(path, 'wt') as f:
                    f.write(pydantic.BaseModel.json(message_exploration, indent=1))
                    f.flush()
                    os.fsync(f.fileno())
                self._stored_explorations[filename] = MessageExplorationStorage(
                    message_exploration=message_exploration.copy(deep=True), last_updated=time.time()
                )
            return False

    def setup_watchdog(self):
        """Set up the observer."""
        # create directory if it does not exist
        if not os.path.exists(self.DIRECTORY):
            os.makedirs(self.DIRECTORY, exist_ok=True)

        self.observer.schedule(self, self.DIRECTORY, recursive=True)
        self.observer.start()
        # call stop() when we exit the program
        atexit.register(self.observer.stop)

    def _reload(self, path):
        """Reload a file from the filesystem (if it's newer)."""
        # get the canonical path relative to the directory
        filename = os.path.relpath(path, self.DIRECTORY)
        # does the file name end with .json?
        if filename.endswith(".json"):
            full_path = os.path.join(self.DIRECTORY, filename)
            # check whether it is different (and newer) than the one in the dict
            # get the last modified time for the file
            last_modified = os.path.getmtime(full_path)
            storage_entry = self._stored_explorations.get(filename, None)
            if storage_entry is not None and storage_entry.last_updated > last_modified:
                # file is older than the one in the dict
                return

            # load the file
            with open(full_path, "rt") as f:
                try:
                    message_exploration = MessageExploration.parse_raw(f.read())
                except pydantic.ValidationError as e:
                    print(f"Error loading file {full_path}: {e}")
                else:
                    if storage_entry is None or message_exploration != storage_entry.message_exploration:
                        # update the dict
                        self._stored_explorations[filename] = MessageExplorationStorage(
                            message_exploration=message_exploration, last_updated=last_modified
                        )
                        # send an event
                        self.send_update_event(None, message_exploration.uid)

    def load_all(self):
        """Load all message explorations from the filesystem."""
        with self.lock:
            # list all files in the directory
            for filename in os.listdir(self.DIRECTORY):
                # does the file name end with .json?
                if filename.endswith(".json"):
                    # load the file
                    full_path = os.path.join(self.DIRECTORY, filename)
                    with open(full_path, "rt") as f:
                        try:
                            message_exploration = MessageExploration.parse_raw(f.read())

                            last_modified = os.path.getmtime(full_path)

                            self._stored_explorations[filename] = MessageExplorationStorage(
                                message_exploration=message_exploration, last_updated=last_modified
                            )
                        except pydantic.ValidationError as e:
                            print(f"Error loading file {full_path}: {e}")

    def on_moved(self, event):
        """Called when a file is moved."""
        if isinstance(event, watchdog.events.FileMovedEvent):
            # convert both paths to relative paths
            src_path = os.path.relpath(event.src_path, self.DIRECTORY)
            dest_path = os.path.relpath(event.dest_path, self.DIRECTORY)

            with self.lock:
                if src_path in self._stored_explorations:
                    message_exploration = self._stored_explorations[src_path]
                    del self._stored_explorations[src_path]
                    self._stored_explorations[dest_path] = message_exploration
                    self._reload(event.dest_path)

    def on_created(self, event):
        """Called when a file is created."""
        if isinstance(event, watchdog.events.FileCreatedEvent):
            with self.lock:
                if os.path.exists(event.src_path):
                    self._reload(event.src_path)

    def on_deleted(self, event):
        """Called when a file is deleted."""
        if isinstance(event, watchdog.events.FileDeletedEvent):
            # convert both paths to relative paths
            src_path = os.path.relpath(event.src_path, self.DIRECTORY)

            with self.lock:
                if src_path in self._stored_explorations and not os.path.exists(event.src_path):
                    print(f"Deleting {src_path}")
                    del self._stored_explorations[src_path]

    def on_modified(self, event):
        """Called when a file is modified."""
        if isinstance(event, watchdog.events.FileModifiedEvent):
            # check that the file still exists
            with self.lock:
                if os.path.exists(event.src_path):
                    self._reload(event.src_path)


# type: ignore
# noinspection PyDunderSlots
class State(pc.State):
    """The app state."""

    __slots__ = [
        '__weakref__',
        'message_exploration',
        'note_editor',
        'drawer_visible',
        'title_text',
    ]

    message_exploration: MessageExploration = MessageExploration()
    note_editor: bool = False
    drawer_visible: bool = False

    title_text: str = ""

    def on_message_exploration_update(self, message_exploration_uid: str):
        """Called when a message exploration is updated by the StoredExplorationsSingleton."""
        if message_exploration_uid != self.message_exploration.uid:
            return

        return self.select_message_exploration(self, message_exploration_uid)  # type: ignore

    def new_message_exploration(self):
        """Create a new message exploration and make it the current one."""
        self._set_message_exploration(self, MessageExploration())  # type: ignore

    def select_message_exploration(self, message_exploration_uid: str):
        """Select a message exploration by its UID and make it the current one."""
        message_exploration = message_explorations_singleton.get(self, message_exploration_uid)
        if message_exploration != self.message_exploration:
            self._set_message_exploration(self, message_exploration)  # type: ignore

    def _update_message_exploration_registry(self):
        message_explorations_singleton.update(self, self.message_exploration)

    @property
    def message_map(self):
        """Get the message map from the current message exploration."""
        return self.message_exploration.message_uid_map

    @pc.var
    def styled_message_thread(self) -> StyledMessageThread:
        """Get the styled message thread from the current message exploration."""
        return self.message_exploration.compute_styled_current_message_thread()

    @pc.var
    def message_exploration_uid(self):
        """Get the UID of the current message exploration."""
        return self.message_exploration.uid

    def cancel_note_editing(self, _: typing.Any):
        """Cancel editing the note."""
        self.note_editor = False
        self.mark_dirty()

    def start_note_editing(self, _: typing.Any):
        """Start editing the note."""
        self.note_editor = True
        self.mark_dirty()

    @pc.var
    def note(self) -> str:
        """Get the note from the current message exploration."""
        return self.message_exploration.note

    def update_note(self, content):
        """Update the note in the current message exploration."""
        self.message_exploration.note = content
        self._update_message_exploration_registry(self)  # type: ignore
        self.note_editor = False
        self.mark_dirty()

    def update_title(self, content):
        """Update the title in the current message exploration."""
        self.message_exploration.title = content
        self._update_message_exploration_registry(self)  # type: ignore
        self.mark_dirty()

    @pc.var
    def get_available_message_explorations(self) -> list[MessageExploration]:
        """Get all available message explorations."""
        return message_explorations_singleton.get_all(self)

    def remove_message_exploration(self, message_exploration_uid: str):
        """Remove a message exploration by its UID."""
        if message_exploration_uid == self.message_exploration.uid:
            # Handle new message exploration differently?
            self._set_message_exploration(self, MessageExploration())  # type: ignore
        message_explorations_singleton.remove(self, message_exploration_uid)
        self.mark_dirty()

    def _set_message_exploration(self, message_exploration: MessageExploration):
        """Set the current message exploration in the main State.

        :param message_exploration: The message exploration to set.

        This mainly ensures that the title and note fields are set correctly.
        """
        if message_exploration != self.message_exploration:
            self.message_exploration = message_exploration
            self.title_text = message_exploration.title
            self.drawer_visible = False
            self.mark_dirty()


class RemoveExplorationModalState(State):
    """The state for the remove exploration modal (to confirm removing a thread).

    TODO: display the whole thread so we can confirm we're removing the right one.
    """

    show_modal: bool = False
    remove_message_exploration_uid: str | None = None

    @pc.var
    def remove_modal_exploration_title(self) -> str:
        """Get the title of the message exploration to remove."""
        if self.remove_message_exploration_uid is not None:
            return message_explorations_singleton.get(self, self.remove_message_exploration_uid).title
        return ""

    def ask_to_remove(self, message_exploration_uid: str):
        """Ask to remove a message exploration by its UID.

        This will show the modal.
        """
        self.remove_message_exploration_uid = message_exploration_uid
        self.show_modal = True
        self.mark_dirty()

    def do_remove(self):
        """Remove the message exploration."""
        if self.remove_message_exploration_uid is not None:
            self.show_modal = False

            # FIXME: EventHandler.__call__ wraps the event args in a JSON one too many times.
            next_event_spec: pc.event.EventSpec = State.remove_message_exploration(None)
            fixed_event_spec = pc.event.EventSpec(
                handler=next_event_spec.handler,
                args=((next_event_spec.args[0][0], self.remove_message_exploration_uid),),
            )
            self.remove_message_exploration_uid = None
            self.mark_dirty()
            return fixed_event_spec

    def cancel(self):
        """Cancel removing the message exploration."""
        self.show_modal = False
        self.remove_message_exploration_uid = None
        self.mark_dirty()


class NavigationState(State):
    """The state for navigating to a message thread.

    TODO: merge this into the main state.
    """

    def go_to_message_thread(self, thread_id, message_uid):
        """Go to a message thread by its ID."""
        self.message_exploration.current_message_thread_uid = thread_id
        self._update_message_exploration_registry(self)
        self.mark_dirty()


class EditableMessageState(State):
    """The state for editing a message."""

    _editable_message: Message | None = None
    fork_mode: bool = False

    @pc.var
    def editable_message_uid(self) -> str | None:
        """Get the UID of the message being edited."""
        if self._editable_message is None:
            return None
        else:
            return self._editable_message.uid

    @pc.var
    def is_editing(self) -> bool:
        """Whether we're editing a message."""
        return self._editable_message is not None

    @typing.no_type_check
    @pc.var
    def styled_editable_message(self) -> StyledMessage | None:
        """Get the styled message being edited."""
        if self._editable_message is None:
            return None
        else:
            return self._editable_message.compute_editable_style()

    def cancel_editing(self):
        """Cancel editing a message."""
        self._editable_message = None
        self.mark_dirty()

    def submit_editing(self):
        """Submit editing a message."""
        if self._editable_message is None:
            return

        if self.fork_mode:
            EditableMessageState._fork_thread(self)
        else:
            EditableMessageState._edit_thread(self)

        self._editable_message = None
        self.mark_dirty()

    def delete_message(self, uid: str):
        """Delete a message."""
        # copy the current message thread
        message_thread = self.message_exploration.current_message_thread.copy()
        message_thread.uid = UID.create()

        # update the message in the message thread
        message_thread.messages = list(message_thread.messages)

        message = self.message_map[uid]

        # remove the message from the message thread
        message_thread.messages.remove(message)

        # update the message exploration
        self.message_exploration.insert_message_thread(message_thread)
        self._update_message_exploration_registry(self)
        self.mark_dirty()

    def enter_editing(self, uid: str):
        """Enter editing a message."""
        self._editable_message = self.message_map[uid].copy()
        self.fork_mode = False
        self.mark_dirty()

    def enter_forking(self, uid: str):
        """Enter forking a message."""
        self._editable_message = self.message_map[uid].copy()
        self.fork_mode = True
        self.mark_dirty()

    def set_editable_message_content(self, content: str):
        """Set the content of the editable message (from the text area)."""
        assert self._editable_message is not None
        self._editable_message.content = content

    def set_editable_message_role(self, role: str):
        """Set the role of the editable message (from the dropdown)."""
        assert self._editable_message is not None
        self._editable_message.role = MessageRole(role)

    def clear_editable_message_error(self):
        """Clear the error on the editable message."""
        assert self._editable_message is not None
        self._editable_message.error = None
        self.mark_dirty()

    def _edit_thread(self):
        original_message = self.message_map[self._editable_message.uid]

        if self._editable_message == original_message:
            return

        # copy the current message thread
        message_thread = self.message_exploration.current_message_thread.copy()
        message_thread.uid = UID.create()

        # update the message in the message thread
        messages = list(message_thread.messages)

        # create a message with a new uid
        new_message = self._editable_message.copy()
        new_message.uid = UID.create()
        new_message.source = new_message.source.copy()
        new_message.source.edited = True

        messages[messages.index(original_message)] = new_message
        message_thread.messages = messages

        # update the message exploration
        self.message_exploration.insert_message_thread(message_thread)
        self._update_message_exploration_registry(self)

    def _fork_thread(self):
        original_message = self.message_map[self._editable_message.uid]
        if original_message == self._editable_message:
            return

        # copy the current message thread
        message_thread = self.message_exploration.current_message_thread.copy()
        message_thread.uid = UID.create()

        # update the message in the message thread
        message_index = message_thread.messages.index(original_message)
        messages = list(message_thread.messages[: message_index + 1])

        # create a message with a new uid
        new_message = self._editable_message.copy()
        new_message.uid = UID.create()
        if new_message.source.model_type is not None:
            new_message.source = new_message.source.copy()
            new_message.source.edited = True
        else:
            new_message.creation_time = time.time()
        new_message.error = None

        messages[-1] = new_message
        message_thread.messages = messages

        # update the message exploration
        self.message_exploration.insert_message_thread(message_thread)
        self._update_message_exploration_registry(self)


class MessageOverviewState(State):
    """State for showing all alternatives (within the subtree) for a given message."""

    current_message_uid: str | None = None

    def show_alternatives(self, message_uid: str):
        """Show the alternatives for the message with the given uid."""
        self.current_message_uid = message_uid

    def hide_alternatives(self, target_message_uid: str):
        """Hide the alternatives for the message with the given uid. The user has selected the given message."""
        styled_current_message: StyledMessage | None = MessageOverviewState._get_current_message(self)
        if styled_current_message is None:
            return
        # find target_message_uid in prev_linked_messages and next_linked_messages
        linked_messages = [
            message
            for message in styled_current_message.prev_linked_messages + styled_current_message.next_linked_messages
            if message.uid == target_message_uid
        ]
        if len(linked_messages) != 0:
            linked_message = linked_messages[0]
            # set the current message thread to the thread of the linked message
            self.message_exploration.current_message_thread_uid = linked_message.thread_uid
            self._update_message_exploration_registry(self)  # type: ignore

        self.current_message_uid = None
        self.mark_dirty()

    def _get_current_message(self) -> StyledMessage | None:
        return self.styled_message_thread.get_message_by_uid(self.current_message_uid)

    @pc.var
    def styled_current_message(self) -> StyledMessage | None:
        """Get the styled current message (for the carousel)."""
        styled_current_message: StyledMessage | None = MessageOverviewState._get_current_message(self)
        if styled_current_message is None:
            return None
        return styled_current_message.compute_editable_style()

    @pc.var
    def styled_previous_messages(self) -> list[StyledMessage]:
        """Get the styled previous messages (for the carousel)."""
        styled_current_message: StyledMessage | None = MessageOverviewState._get_current_message(self)
        if styled_current_message is None:
            return []
        return [
            self.message_map[prev_linked_message.uid].compute_editable_style()
            for prev_linked_message in styled_current_message.prev_linked_messages
        ]

    @pc.var
    def styled_next_messages(self) -> list[StyledMessage]:
        """Get the styled next messages (for the carousel)."""
        styled_current_message: StyledMessage | None = MessageOverviewState._get_current_message(self)
        if styled_current_message is None:
            return []
        return [
            self.message_map[next_linked_message.uid].compute_editable_style()
            for next_linked_message in styled_current_message.next_linked_messages
        ]


class MessageGridState(State):
    """State for the message thread overview."""

    _show_grid: bool = False

    # @property
    # def _grid(self) -> MessageGrid | None:
    #     if not self._show_grid:
    #         return None
    #     return

    def remove_thread(self, thread_uid: str):
        """Remove the thread with the given uid from the message exploration."""
        self.message_exploration.delete_message_thread(thread_uid)
        self._update_message_exploration_registry(self)  # type: ignore
        self.mark_dirty()

    def toggle_grid(self):
        """Toggle the visibility of the overview map."""
        if not self._show_grid:
            self._show_grid = True
        else:
            self._show_grid = False
        self.mark_dirty()

    def go_to_thread(self, thread_uid: str):
        """Go to the thread with the given uid and make it the current one."""
        self.message_exploration.current_message_thread_uid = thread_uid
        self._update_message_exploration_registry(self)  # type: ignore
        self._show_grid = False
        self.mark_dirty()

    @pc.var
    def is_grid_visible(self) -> bool:
        """Whether the overview map is visible."""
        return self._show_grid

    @pc.var
    def column_thread_uids(self) -> list[str]:
        """The uids of the threads in the overview map."""
        if not self._show_grid:
            return []
        return self.message_exploration.compute_message_grid().thread_uids

    @pc.var
    def styled_grid_cells(self) -> list[list[StyledGridCell]]:
        """The cells of the overview map with the styled messages in them."""
        if not self._show_grid:
            return [[]]

        grid = self.message_exploration.compute_message_grid()

        styled_grid_cells = []
        for row_idx, messages in enumerate(grid.message_grid):
            styled_row = []
            for col_idx, message in enumerate(messages):
                styled_grid_cell = StyledGridCell(
                    message=message.compute_editable_style() if message is not None else None,
                    row_idx=row_idx + 1,
                    col_idx=col_idx + 1,
                    thread_uid=grid.thread_uids[col_idx],
                )
                styled_row.append(styled_grid_cell)
            styled_grid_cells.append(styled_row)

        # compress the grid row-by-row using col_span for duplicated cells
        for row_idx, row in enumerate(styled_grid_cells):
            col_idx = 0
            while col_idx < len(row):
                cell = row[col_idx]
                if cell.message is None:
                    col_idx += 1
                    continue
                col_span = 1
                while col_idx + col_span < len(row) and row[col_idx + col_span].message == cell.message:
                    col_span += 1
                if col_span > 1:
                    row[col_idx].col_span = col_span
                    for i in range(col_idx + 1, col_idx + col_span):
                        row[i].skip = True
                col_idx += col_span
        return styled_grid_cells


async def send_event(state: pc.State, event_handler: pc.event.EventHandler):
    """Send an event to the server (from a different client thread)."""
    events = pc.event.fix_events([event_handler], state.get_token())
    state_update = pc.state.StateUpdate(delta={}, events=events)
    state_update_json = state_update.json()

    await app.sio.emit(str(pc.constants.SocketEvent.EVENT), state_update_json, to=state.get_sid(), namespace="/event")


class PartialResponsePayload(typing.TypedDict):
    """The payload of a partial response from the receiver thread."""

    uid: str
    content: str


def run_coroutine_in_thread(coroutine, *args, **kwargs):
    """Run a coroutine in a separate thread.

    TODO(black): this is part of a workaround to get around the fact that the events are being dropped when using
        app.sio.emit.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    asyncio.run(coroutine(*args, **kwargs))


def helper_thread(coroutine, *args, **kwargs):
    """Trying to fix issues with streaming messages.

    Events keep getting dropped, so I've tried various things to fix it.

    This is the latest attempt, which is to run the coroutine in a separate thread and use a barrier
    and minimum delays. Sadly, this is a lot more of a hack than I'd like.
    """
    t = threading.Thread(target=run_coroutine_in_thread, args=(coroutine, *args), kwargs=kwargs)
    t.start()


class ModelRequestsState(State):
    """The state for LLM requests."""

    # TODO: this is a hack to get around the fact that the events are being dropped when using app.sio.emit.
    _request_barrier = threading.Barrier(2)

    model_type: str = "GPT-3.5"
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    _active_message: Message | None = None
    user_message_text: str = ""

    @pc.var
    def active_generation(self) -> bool:
        """Whether an LLM request is currently active."""
        return self._active_message is not None

    @pc.var
    def styled_active_message(self) -> StyledMessage | None:
        """The active message with the styled message.

        TODO(blackhc): remove this again (because we use the regular display for this now?)
        """
        if self._active_message is None:
            return None
        return self._active_message.compute_editable_style()

    @pc.var
    def chat_mode(self) -> bool:
        """Whether the user is in chat mode.

        We want to display a message window at the bottom.
        """
        messages = self.message_exploration.current_message_thread.messages
        if len(messages) == 0 or messages[-1].role != MessageRole.HUMAN:
            return True
        return False

    def cancel_active_chat_completion(self):
        """Cancel the active request as a user action."""
        if self._active_message is None:
            return
        self._active_message.error = "Canceled by user."
        self._active_message = None
        self._update_message_exploration_registry(self)
        self.mark_dirty()

    def receive_partial_response(self, payload_str: str):
        """Receive a partial response from the receiver thread.

        We currently use a barrier to sync the different threads to avoid messages being dropped.

        Maybe adding a delay of 50ms was already enough, but I don't trust it anymore and don't want to have any
        dropped words anymore.

        TODO(blackhc): split a receive_finished_response method out of this (content="None" means finished).
        """
        # parse payload_str into a PartialResponsePayload using json.loads
        payload: PartialResponsePayload = json.loads(payload_str)

        if self._active_message is None or self._active_message.uid != payload['uid']:
            return

        if payload['content'] is None:
            self._active_message = None
            self._update_message_exploration_registry(self)
        else:
            assert isinstance(payload['content'], str)
            assert self._active_message.content is not None
            self._active_message.content += payload['content']

            last_updated = message_explorations_singleton.get_last_updated_timestamp(self.message_exploration_uid)
            # This seems to slow down the app too much
            if last_updated + 1 < time.time():
                self._update_message_exploration_registry(self)  # type: ignore
        self.mark_dirty()

        # Signal to the other thread that we're done. (If this times out, that's not bad by itself.)
        try:
            self._request_barrier.wait(timeout=0.1)
        except threading.BrokenBarrierError:
            pass

    def receive_error(self, payload_str: str):
        """Receive an error from the receiver thread."""
        # parse payload_str into a PartialResponsePayload using json.loads
        payload: PartialResponsePayload = json.loads(payload_str)

        if self._active_message is None or self._active_message.uid != payload['uid']:
            return

        self._active_message.error = payload['content']
        self._active_message = None
        self._update_message_exploration_registry(self)  # type: ignore
        self.mark_dirty()

    async def _query_openai(self, active_message, openai_messages):
        try:
            response = await active_message.source.query_openai(openai_messages)
            residual_content = ""
            async with aclosing(response):
                async for partial_response in response:
                    if self._active_message is None or self._active_message.uid != active_message.uid:
                        return

                    # finish reason?
                    if partial_response['choices'][0]['finish_reason'] == 'stop':
                        break

                    delta = partial_response['choices'][0]['delta']

                    if 'role' in delta:
                        assert delta['role'] == 'assistant'

                    if 'content' not in delta:
                        continue

                    content = residual_content + delta['content']
                    # noinspection PyNoneFunctionAssignment,PyArgumentList
                    event_handler: pc.event.EventHandler = ModelRequestsState.receive_partial_response(  # type: ignore
                        PartialResponsePayload(
                            uid=active_message.uid,
                            content=content,
                        )
                    )
                    await send_event(self, event_handler)

                    try:
                        self._request_barrier.wait(timeout=1.0)
                        residual_content = ""
                    except threading.BrokenBarrierError:
                        residual_content += content
                        self._request_barrier = threading.Barrier(2)

                    # Minimum wait time to avoid dropped packets.
                    await asyncio.sleep(0.01)

            # Make sure any residual content is still sent.
            while residual_content:
                if self._active_message is None or self._active_message.uid != active_message.uid:
                    return
                # noinspection PyNoneFunctionAssignment,PyArgumentList
                event_handler: pc.event.EventHandler = ModelRequestsState.receive_partial_response(  # type: ignore
                    PartialResponsePayload(
                        uid=active_message.uid,
                        content=content,
                    )
                )
                await send_event(self, event_handler)

                try:
                    self._request_barrier.wait(timeout=1.0)
                except threading.BrokenBarrierError:
                    self._request_barrier = threading.Barrier(2)
                else:
                    break

                await asyncio.sleep(0.01)

            # noinspection PyNoneFunctionAssignment,PyArgumentList
            event_handler: pc.event.EventHandler = ModelRequestsState.receive_partial_response(  # type: ignore
                PartialResponsePayload(
                    uid=active_message.uid,
                    content=None,
                )
            )
            while True:
                await send_event(self, event_handler)
                try:
                    self._request_barrier.wait(timeout=1.0)
                except threading.BrokenBarrierError:
                    pass
                else:
                    break

                await asyncio.sleep(0.01)
        except asyncio.TimeoutError:
            # noinspection PyNoneFunctionAssignment,PyArgumentList
            event_handler: pc.event.EventHandler = ModelRequestsState.receive_error(  # type: ignore
                PartialResponsePayload(
                    uid=active_message.uid,
                    content="Request Timeout Error",
                )
            )
            await send_event(self, event_handler)
        except Exception as e:
            # noinspection PyNoneFunctionAssignment,PyArgumentList
            event_handler: pc.event.EventHandler = ModelRequestsState.receive_error(  # type: ignore
                PartialResponsePayload(
                    uid=active_message.uid,
                    content=str(e),
                )
            )
            await send_event(self, event_handler)
            raise

    def _request_chat_model_completion(self):
        """Helper function to request a chat model completion."""
        active_message = Message(
            uid=UID.create(),
            role=MessageRole.BOT,
            content="",
            creation_time=time.time(),
            source=MessageSource(
                model_type=self.model_type,
                model_dict={
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    "frequency_penalty": self.frequency_penalty,
                    "presence_penalty": self.presence_penalty,
                },
            ),
        )

        self._active_message = active_message
        self.message_exploration.add_message(active_message)
        self._update_message_exploration_registry(self)

        openai_messages = [
            message.get_openai_json() for message in self.message_exploration.current_message_thread.messages
        ]

        helper_thread(self._query_openai, self, active_message=active_message, openai_messages=openai_messages)

    def request_chat_model_completion(self):
        """Request a chat model completion (user action)."""
        self._request_chat_model_completion(self)

    def submit_message(self):
        """Submit a user message and then ask for a chat model completion (user action)."""
        user_message = Message(
            uid=UID.create(),
            role=MessageRole.HUMAN,
            content=self.user_message_text,
            creation_time=time.time(),
            source=MessageSource(
                model_type=None,
            ),
        )
        self.message_exploration.add_message(user_message)
        self._update_message_exploration_registry(self)

        self.user_message_text = ""
        self.mark_dirty()

        self._request_chat_model_completion(self)


def index() -> pc.Component:
    """Render the index page."""
    return pc.container(
        pc.vstack(
            render_message_exploration(State.styled_message_thread),
            width="100",
        ),
        render_app_drawer(),
        padding_top="64px",
        padding_bottom="64px",
        max_width="80ch",
    )


message_explorations_singleton = StoredExplorationsSingleton()

meta = [
    {"name": "theme_color", "content": "#FFFFFF"},
    {"char_set": "UTF-8"},
    # {"property": "og:url", "content": "url"},
]

# Add state and page to the app.
app = pc.App(
    state=State,
    stylesheets=[
        'react-json-view-lite.css',
    ],
)
app.add_page(
    index,
    meta=meta,
    title="ChatPlayground",
    description="Local Chat Playground for LLMs",
)
app.compile()
